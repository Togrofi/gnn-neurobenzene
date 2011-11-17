//----------------------------------------------------------------------------
/** @file VCS.cpp */
//----------------------------------------------------------------------------

#include "SgSystem.h"
#include "SgTimer.h"

#include "Hex.hpp"
#include "BitsetIterator.hpp"
#include "Misc.hpp"
#include "VCS.hpp"
#include "VCOr.hpp"
#include "VCPattern.hpp"

using namespace benzene;

//----------------------------------------------------------------------------

VCBuilderParam::VCBuilderParam()
    : and_over_edge(false),
      use_patterns(true),
      use_non_edge_patterns(true),
      incremental_builds(true),
      threats(true)
{
}

//----------------------------------------------------------------------------

inline CarrierList::CarrierList()
{
}

inline CarrierList::CarrierList(bitset_t carrier)
    : m_list(1, carrier)
{
}

inline CarrierList::CarrierList(const std::vector<bitset_t>& carriers_list)
{
    m_list.reserve(carriers_list.size());
    for (std::size_t i = 0; i < carriers_list.size(); ++i)
        AddNew(carriers_list[i]);
}

inline void CarrierList::AddNew(bitset_t carrier)
{
    m_list.push_back(carrier);
}

inline bool CarrierList::SupersetOfAny(bitset_t carrier) const
{
    for (std::size_t i = 0; i < m_list.size(); ++i)
        if (BitsetUtil::IsSubsetOf(m_list[i].carrier, carrier))
        {
            // Move to front
            Elem c = m_list[i];
            for (std::size_t j = i; j > 0; j--)
                m_list[j] = m_list[j - 1];
            m_list[0] = c;
            return true;
        }
    return false;
}

inline bool CarrierList::RemoveSupersetsOfCheckAnyRemoved(const CarrierList& filter)
{
    std::size_t j = 0;
    std::size_t i = 0;
    for (; i < m_list.size(); ++i)
    {
        if (!filter.SupersetOfAny(m_list[i].carrier))
            m_list[j++] = m_list[i];
    }
    m_list.resize(j);
    return j < i;
}

template <bool check_old>
inline bool CarrierList::RemoveSupersetsOf(bitset_t carrier)
{
    std::size_t j = 0;
    std::size_t i = 0;
    bool check_old_res = false;
    for (; i < m_list.size(); ++i)
    {
        if (!BitsetUtil::IsSubsetOf(carrier, m_list[i].carrier))
            m_list[j++] = m_list[i];
        else if (check_old)
        {
            if (m_list[i].old)
                check_old_res = true;
        }
    }
    m_list.resize(j);
    if (check_old)
        return check_old_res;
    else
        return j < i;
}

template <bool store_removed>
inline size_t CarrierList::RemoveAllContaining_(bitset_t set,
                                                std::vector<bitset_t>* removed)
{
    std::size_t j = 0;
    std::size_t i = 0;
    for (; i < m_list.size(); ++i)
    {
        if ((set & m_list[i].carrier).none())
            m_list[j++] = m_list[i];
        else if (store_removed)
            removed->push_back(m_list[i].carrier);
    }
    m_list.resize(j);
    return i - j;
}

inline size_t CarrierList::RemoveAllContaining(bitset_t set)
{
    return RemoveAllContaining_<false>(set, 0);
}

inline size_t CarrierList::RemoveAllContaining(bitset_t set,
                                             std::vector<bitset_t>& removed)
{
    return RemoveAllContaining_<true>(set, &removed);
}

inline bool CarrierList::RemoveSupersetsOfCheckOldRemoved(bitset_t carrier)
{
    return RemoveSupersetsOf<true>(carrier);
}

inline bool CarrierList::RemoveSupersetsOfCheckAnyRemoved(bitset_t carrier)
{
    return RemoveSupersetsOf<false>(carrier);
}

inline void CarrierList::RemoveSupersetsOfUnchecked(bitset_t carrier)
{
    RemoveSupersetsOf<false>(carrier);
}

inline bool CarrierList::TrySetOld(bitset_t carrier) const
{
    for (std::size_t i = 0; i < m_list.size(); ++i)
        if (m_list[i].carrier == carrier)
        {
            m_list[i].old = true;
            return true;
        }
    return false;
}

bitset_t CarrierList::GetGreedyUnion() const
{
    bitset_t U;
    bitset_t I;
    I.set();
    for (Iterator i(*this); i; ++i)
    {
        if ((I & i.Carrier()) != I)
        {
            I &= i.Carrier();
            U |= i.Carrier();
        }
    }
    return U;
}

template <bool only_old>
inline bitset_t CarrierList::GetIntersection() const
{
    bitset_t I;
    I.set();
    for (Iterator i(*this); i; ++i)
        if (!only_old || i.Old())
            I &= i.Carrier();
    return I;
}

inline bitset_t CarrierList::GetOldIntersection() const
{
    return GetIntersection<true>();
}

inline bitset_t CarrierList::GetAllIntersection() const
{
    return GetIntersection<false>();
}

inline void CarrierList::MarkAllOld()
{
    for (std::size_t i = 0; i < m_list.size(); i++)
        m_list[i].old = true;
}

inline void CarrierList::MarkAllNew()
{
    for (std::size_t i = 0; i < m_list.size(); i++)
        m_list[i].old = false;
}

inline void CarrierList::Clear()
{
    m_list.clear();
}

//----------------------------------------------------------------------------

inline VCS::Ends::Ends(HexPoint x, HexPoint y)
    : x(x), y(y)
{
}

inline VCS::Full::Full(HexPoint x, HexPoint y, bitset_t carrier)
    : Ends(x, y), carrier(carrier)
{
}

inline VCS::Semi::Semi(HexPoint x, HexPoint y, bitset_t carrier, HexPoint key)
    : Full(x, y, carrier), key(key)
{
}

//----------------------------------------------------------------------------

inline VCS::AndList::AndList()
{
    m_processed_intersection.set();
}

inline VCS::AndList::AndList(bitset_t carrier)
    : CarrierList(carrier)
{
    m_processed_intersection.set();
}

inline VCS::AndList::AndList(const std::vector<bitset_t>& carriers_list)
    : CarrierList(carriers_list)
{
    m_processed_intersection.set();
}

inline void VCS::AndList::RemoveSupersetsOf(bitset_t carrier)
{
    if (RemoveSupersetsOfCheckOldRemoved(carrier))
        CalcIntersection();
}

inline void VCS::AndList::Add(bitset_t carrier)
{
    RemoveSupersetsOf(carrier);
    AddNew(carrier);
}

inline bool VCS::AndList::TryAdd(bitset_t carrier)
{
    if (SupersetOfAny(carrier))
        return false;
    Add(carrier);
    return true;
}

inline bool VCS::AndList::TryAdd(bitset_t carrier, const CarrierList& filter)
{
    if (filter.SupersetOfAny(carrier))
        return false;
    else
        return TryAdd(carrier);
}

inline bitset_t VCS::AndList::GetIntersection() const
{
    return m_processed_intersection;
}

inline bool VCS::AndList::TrySetProcessed(bitset_t carrier)
{
    if (TrySetOld(carrier))
    {
        m_processed_intersection &= carrier;
        return true;
    }
    else
        return false;
}

inline void VCS::AndList::MarkAllUnprocessed()
{
    MarkAllNew();
    CalcIntersection();
}

inline void VCS::AndList::CalcIntersection()
{
    m_processed_intersection = GetOldIntersection();
}

//----------------------------------------------------------------------------

inline VCS::OrList::OrList()
    : m_queued(false)
{
    m_intersection.set();
}

inline VCS::OrList::OrList(bitset_t carrier)
    : CarrierList(carrier),
      m_intersection(carrier),
      m_queued(false)
{
}

inline VCS::OrList::OrList(const CarrierList& carrier_list,
                           bitset_t intersection)
    : CarrierList(carrier_list),
      m_intersection(intersection),
      m_queued(false)
{
}

inline bitset_t VCS::OrList::GetIntersection() const
{
    return m_intersection;
}

inline void VCS::OrList::Add(bitset_t carrier)
{
    if (SupersetOfAny(carrier))
        return;
    RemoveSupersetsOfUnchecked(carrier);
    AddNew(carrier);
    m_intersection &= carrier;
}

inline bool VCS::OrList::RemoveSupersetsOf(bitset_t carrier)
{
    if (RemoveSupersetsOfCheckAnyRemoved(carrier))
    {
        m_intersection = GetAllIntersection();
        return true;
    }
    else
        return false;
}

inline bool VCS::OrList::TryQueue(bitset_t capturedSet)
{
    bool prev_queued = m_queued;
    m_queued = BitsetUtil::IsSubsetOf(m_intersection, capturedSet);
    return !prev_queued && m_queued;
}

void VCS::OrList::MarkAllProcessed()
{
    MarkAllOld();
    m_queued = false;
}

void VCS::OrList::MarkAllUnprocessed()
{
    MarkAllNew();
}

void VCS::OrList::Clear()
{
    CarrierList::Clear();
    m_intersection.set();
}

//---------------------------------------------------------------------------

inline VCS::SemiList::SemiList()
{
}

inline VCS::SemiList::SemiList(const CarrierList& carrier_list, bitset_t intersection)
    : OrList(carrier_list, intersection)
{
}

inline VCS::SemiList::SemiList(bitset_t carrier, HexPoint key)
    : OrList(carrier)
{
    Put(key, new AndList(carrier));
}

inline void VCS::SemiList::RemoveSupersetsOf(bitset_t carrier)
{
    if (!OrList::RemoveSupersetsOf(carrier))
        return;
    for (BitsetIterator it(Entries()); it; ++it)
        (*this)[*it]->RemoveSupersetsOf(carrier);
}

inline bool VCS::SemiList::RemoveSupersetsOf(const CarrierList& filter)
{
    bool res = false;
    for (BitsetIterator it(Entries()); it; ++it)
        res |= (*this)[*it]->RemoveSupersetsOfCheckAnyRemoved(filter);
    return res;
}

void VCS::SemiList::CalcAllSemis()
{
    Clear();
    for (BitsetIterator k(Entries()); k; ++k)
        for (CarrierList::Iterator i(*(*this)[*k]); i; ++i)
            Add(i.Carrier());
}

//---------------------------------------------------------------------------

inline void VCS::TestQueuesEmpty()
{
    BenzeneAssert(m_fulls_and_queue.IsEmpty());
    BenzeneAssert(m_semis_and_queue.IsEmpty());
    BenzeneAssert(m_semis_or_queue.IsEmpty());
}

VCS::VCS(HexColor color)
    : m_color(color),
      m_edge1(HexPointUtil::colorEdge1(color)),
      m_edge2(HexPointUtil::colorEdge2(color))
{
    LoadCapturedSetPatterns();
}

void VCS::LoadCapturedSetPatterns()
{
    std::ifstream inFile;
    try {
        std::string file = MiscUtil::OpenFile("vc-captured-set.txt", inFile);
        LogConfig() << "VCS: reading captured set patterns from '"
        << file << "'.\n";
    }
    catch (BenzeneException& e) {
        throw BenzeneException() << "VCS: " << e.what();
    }
    std::vector<Pattern> patterns;
    Pattern::LoadPatternsFromStream(inFile, patterns);
    LogConfig() << "VCS:: parsed " << patterns.size() << " patterns.\n";
    for (std::size_t i = 0; i < patterns.size(); ++i)
    {
        m_capturedSetPatterns[WHITE].push_back(patterns[i]);
        patterns[i].FlipColors();
        m_capturedSetPatterns[BLACK].push_back(patterns[i]);
    }
    for (BWIterator c; c; ++c)
        m_hash_capturedSetPatterns[*c].Hash(m_capturedSetPatterns[*c]);
}

VCS::VCS(const VCS& other)
    : m_brd(other.m_brd),
      m_color(other.m_color)
{
    BenzeneAssert(false /* TODO implement copy constructor */);
}

/** Computes the 0-connections defined by adjacency.*/
void VCS::AddBaseVCs()
{
    HexColorSet not_other = HexColorSetUtil::ColorOrEmpty(m_color);
    for (GroupIterator x(*m_groups, not_other); x; ++x)
    {
        for (BitsetIterator y(x->Nbs() & m_brd->GetEmpty()); y; ++y)
        {
            BenzeneAssert(*y == m_groups->CaptainOf(*y));
            m_statistics.base_attempts++;
            BenzeneAssert(x->Captain() != *y);
            if (TryAddFull(x->Captain(), *y, EMPTY_BITSET))
                m_statistics.base_successes++;
        }
    }
}

/** Adds vcs obtained by pre-computed patterns. */
void VCS::AddPatternVCs()
{
    const VCPatternSet& patterns
    = VCPattern::GetPatterns(m_brd->Width(), m_brd->Height(), m_color);
    for (std::size_t i=0; i<patterns.size(); ++i)
    {
        const VCPattern& pat = patterns[i];
        if (!m_param->use_non_edge_patterns
            && !HexPointUtil::isEdge(pat.Endpoint(0))
            && !HexPointUtil::isEdge(pat.Endpoint(1)))
            continue;
        if (pat.Matches(m_color, *m_brd))
        {
            bitset_t carrier = pat.NotOpponent() - m_brd->GetColor(m_color);
            carrier.reset(pat.Endpoint(0));
            carrier.reset(pat.Endpoint(1));

            HexPoint x = m_groups->CaptainOf(pat.Endpoint(0));
            HexPoint y = m_groups->CaptainOf(pat.Endpoint(1));
            if (x == y)
                continue;

            m_statistics.pattern_attempts++;
            if (TryAddFull(x, y, carrier))
                m_statistics.pattern_successes++;
        }
    }
}

void VCS::ComputeCapturedSets(const PatternState& patterns)
{
    SG_UNUSED(patterns);
    for (BoardIterator p(m_brd->Const().EdgesAndInterior()); p; ++p)
    {
        m_capturedSet[*p] = EMPTY_BITSET;
        if (m_brd->GetColor(*p) == EMPTY)
        {
            PatternHits hits;
            patterns.MatchOnCell(m_hash_capturedSetPatterns[m_color],
                                 *p, PatternState::STOP_AT_FIRST_HIT, hits);
            for (std::size_t i = 0; i < hits.size(); ++i)
            {
                const std::vector<HexPoint>& moves = hits[0].Moves2();
                for (std::size_t j = 0; j < moves.size(); ++j)
                    m_capturedSet[*p].set(moves[j]);
            }
        }
    }
}

void VCS::Build(VCBuilderParam& param,
                const Groups& groups, const PatternState& patterns)
{
    TestQueuesEmpty();
    SgTimer timer;
    m_param = &param;
    m_groups = &groups;
    m_brd = &m_groups->Board();
    Reset();

    ComputeCapturedSets(patterns);
    AddBaseVCs();
    if (m_param->use_patterns)
        AddPatternVCs();
    DoSearch();

    LogFine() << "  " << timer.GetTime() << "s to build vcs.\n";
}

void VCS::Build(VCBuilderParam& param,
                const Groups& oldGroups, const Groups& newGroups,
                const PatternState& patterns,
                bitset_t added[BLACK_AND_WHITE], bool use_changelog)
{
    TestQueuesEmpty();
    SgTimer timer;
    m_param = &param;
    m_groups = &newGroups;
    m_brd = &m_groups->Board();
    if (use_changelog)
    {
        backups.push_back(Backup());
        backups.back().Create(*this);
    }

    if (param.incremental_builds)
    {
        bitset_t capturedSet[BITSETSIZE];
        for (int i = 0; i < BITSETSIZE; i++)
            capturedSet[i] = m_capturedSet[i];
        m_statistics = Statistics();
        ComputeCapturedSets(patterns);
        Merge(oldGroups, capturedSet, added);
    }
    else
    {
        Reset();
        ComputeCapturedSets(patterns);
        AddBaseVCs();
    }

    if (m_param->use_patterns)
        AddPatternVCs();
    DoSearch();

    LogFine() << "  " << timer.GetTime() << "s to build vcs.\n";
}

void VCS::Revert()
{
    Reset();
    backups.back().Restore(*this);
    backups.pop_back();
}

void VCS::Reset()
{
    m_fulls.Reset();
    m_semis.Reset();

    m_statistics = Statistics();
}

/** @page mergeshrink Incremental Update Algorithm

    The connection set is updated to the new state of the board in a
    single pass. In this pass connections touched by opponent stones
    are destroyed, connections touched by friendly stones are resized,
    and connections in groups that are merged into larger groups are
    merged into the proper connection lists. This entire process is
    called the "merge".

    Every list needs to be checked for shrinking. This entails
    removing any cells from a connection's carrier that are now
    occupied by friendly stones. Semi-connections that have their keys
    played must be upgraded to full connections. */
void VCS::Merge(const Groups& oldGroups, bitset_t* oldCapturedSet,
                bitset_t added[BLACK_AND_WHITE])
{
    // Kill connections containing stones the opponent just played.
    // NOTE: This *must* be done in the original state, not in the
    // state with the newly added stones. If we are adding stones of
    // both colors there could be two groups of our stones that are
    // going to be merged, but we need to kill connections touching
    // the opponent stones before we do so.
    RemoveAllContaining(oldGroups, added[!m_color]);

    bitset_t merged[BITSETSIZE];
    HexColorSet not_other = HexColorSetUtil::NotColor(!m_color);
    for (GroupIterator x(oldGroups, not_other); x; ++x)
    {
        HexPoint xc = x->Captain();
        merged[m_groups->CaptainOf(xc)].set(xc);
    }

    for (GroupIterator x(*m_groups, not_other); x; ++x)
    {
        HexPoint xc = x->Captain();
        MergeRemoveSelfEnds(merged[xc]);
        for (GroupIterator y(*m_groups, not_other); &*y != &*x; ++y)
        {
            HexPoint yc = y->Captain();
            MergeAndShrink(oldCapturedSet, added[m_color],
                           merged[xc], merged[yc], xc, yc);
        }
    }
}

/** Removes all connections whose intersection with given set is
    non-empty. Any list that is modified is added to the queue, since
    some unprocessed connections could have been brought under the
    softlimit. */
inline void VCS::RemoveAllContaining(const Groups& oldGroups,
                                     bitset_t removed)
{
    // Use old groupset and delete old groups that are
    // now the opponent's color.
    HexColorSet not_other = HexColorSetUtil::NotColor(!m_color);
    for (GroupIterator x(oldGroups, not_other); x; ++x)
    {
        HexPoint xc = x->Captain();
        bool killed = m_groups->GetGroup(xc).Color() == !m_color;
        for (GroupIterator y(oldGroups, not_other); &*y != &*x; ++y)
        {
            HexPoint yc = y->Captain();
            if (killed || m_groups->GetGroup(yc).Color() == !m_color)
            {
                m_fulls.Delete(xc, yc);
                m_semis.Delete(xc, yc);
                continue;
            }
            AndList* fulls = m_fulls[xc][yc];
            SemiList* semis = m_semis[xc][yc];
            if (fulls)
            {
                m_statistics.killed0 += fulls->RemoveAllContaining(removed);
                if (fulls->IsEmpty())
                    m_fulls.Delete(xc, yc, fulls);
            }
            if (semis)
            {
                size_t total_removed = 0;
                for (BitsetIterator k(semis->Entries()); k; ++k)
                {
                    AndList* key_semis = (*semis)[*k];
                    size_t r = key_semis->RemoveAllContaining(removed);
                    m_statistics.killed1 += r;
                    total_removed += r;
                    if (key_semis->IsEmpty())
                        semis->Delete(*k);
                }
                if (semis->Entries().none())
                    m_semis.Delete(xc, yc, semis);
                else if (total_removed)
                    semis->CalcAllSemis();
            }
        }
    }
}

inline void VCS::MergeRemoveSelfEnds(bitset_t x_merged)
{
    for (BitsetIterator x(x_merged); x; ++x)
        for (BitsetIterator y(x_merged); *y <= *x; ++y)
        {
            m_fulls.Delete(*x, *y);
            m_semis.Delete(*x, *y);
        }
}

inline bool VCS::Shrink(bitset_t added, HexPoint x, HexPoint y,
                        AndList* fulls)
{
    bool new_fulls = false;
    std::vector<bitset_t> to_shrink;
    fulls->RemoveAllContaining(added, to_shrink);
    for (std::vector<bitset_t>::iterator it = to_shrink.begin();
         it != to_shrink.end(); ++it)
    {
        bitset_t carrier = *it - added;
        if (fulls->TryAdd(carrier))
        {
            m_statistics.shrunk0++;
            m_fulls_and_queue.Push(Full(x, y, carrier));
            new_fulls = true;
        }
    }
    return new_fulls;
}

inline bool VCS::Shrink(bitset_t added, HexPoint x, HexPoint y,
                        AndList* semis,
                        const AndList* filter, HexPoint key)
{
    bool new_semis = false;
    std::vector<bitset_t> to_shrink;
    semis->RemoveAllContaining(added, to_shrink);
    for (std::vector<bitset_t>::iterator it = to_shrink.begin();
         it != to_shrink.end(); ++it)
         {
             bitset_t carrier = *it - added;
             bool success;
             if (filter)
                 success = semis->TryAdd(carrier, *filter);
             else
                 success = semis->TryAdd(carrier);
             if (success)
             {
                 m_statistics.shrunk1++;
                 m_semis_and_queue.Push(Semi(x, y, carrier, key));
                 new_semis = true;
             }
         }
         return new_semis;
}

inline bool VCS::Shrink(bitset_t added, HexPoint x, HexPoint y,
                        AndList* fulls, const CarrierList& list,
                        size_t& stats)
{
    bool new_fulls = false;
    for (CarrierList::Iterator i(list); i; ++i)
    {
        bitset_t carrier = i.Carrier() - added;
        if (fulls->TryAdd(carrier))
        {
            stats++;
            m_fulls_and_queue.Push(Full(x, y, carrier));
            new_fulls = true;
        }
    }
    return new_fulls;
}

inline bool VCS::Shrink(bitset_t added, HexPoint x, HexPoint y,
                        AndList* semis, const CarrierList& list,
                        const AndList* filter, HexPoint key)
{
    bool new_semis = false;
    for (CarrierList::Iterator i(list); i; ++i)
    {
        bitset_t carrier = i.Carrier() - added;
        bool success;
        if (filter)
            success = semis->TryAdd(carrier, *filter);
        else
            success = semis->TryAdd(carrier);
        if (success)
        {
            m_statistics.shrunk1++;
            m_semis_and_queue.Push(Semi(x, y, carrier, key));
            new_semis = true;
        }
    }
    return new_semis;
}

/** Merges and shrinks connections between the given endpoints. */
void VCS::MergeAndShrink(bitset_t* oldCapturedSet, bitset_t added,
                         bitset_t x_merged, bitset_t y_merged,
                         HexPoint x, HexPoint y)
{
    BenzeneAssert(x != y);
    BenzeneAssert((x_merged & y_merged).none());

    AndList* fulls = m_fulls[x][y];
    SemiList* semis = m_semis[x][y];

    bool new_fulls = false;
    bool newCaptureSet =
        (oldCapturedSet[x] | oldCapturedSet[y]) !=
        (m_capturedSet[x] | m_capturedSet[y]);

    if (fulls)
        // First shrink fulls wihtout merge
        new_fulls |= Shrink(added, x, y, fulls);

    // Collect and shrink all fulls including upgraded semis
    for (BitsetIterator xi(x_merged); xi; ++xi)
        for (BitsetIterator yi(y_merged); yi; ++yi)
        {
            // Merge and shrink fulls with merged ends
            if (*xi != x || *yi != y)
            {
                AndList* merged_fulls = m_fulls[*xi][*yi];
                if (merged_fulls)
                {
                    if (!fulls)
                        fulls = m_fulls.Put(x, y);
                    new_fulls |= Shrink(added, x, y, fulls, *merged_fulls,
                                        m_statistics.shrunk0);
                    m_fulls.Delete(*xi, *yi, merged_fulls);
                }
            }

            // Upgrade semis
            SemiList* merged_semis = m_semis[*xi][*yi];
            if (!merged_semis)
                continue;
            for (BitsetIterator k(merged_semis->Entries() & added); k; ++k)
            {
                AndList* key_semis = (*merged_semis)[*k];
                if (!fulls)
                    fulls = m_fulls.Put(x, y);
                new_fulls |= Shrink(added, x, y, fulls, *key_semis,
                                    m_statistics.upgraded);
                merged_semis->Delete(*k);
            }
        }

    if (fulls)
    {
        if (added.test(x) || added.test(y) || newCaptureSet)
        {
            for (CarrierList::Iterator i(*fulls); i; ++i)
                if (i.Old())
                    m_fulls_and_queue.Push(Full(x, y, i.Carrier()));
                fulls->MarkAllUnprocessed();
        }
        else
            fulls->CalcIntersection();
    }

    // Shrink semis
    bool calc_all_semis = false;
    if (semis)
    {
        if (new_fulls)
            // Filter out semis which are supersets of new fulls
            calc_all_semis |= semis->RemoveSupersetsOf(*fulls);
        // Shrink semis without merge
        for (BitsetIterator k(semis->Entries()); k; ++k)
            calc_all_semis |= Shrink(added, x, y, (*semis)[*k], fulls, *k);
    }

    // Collect and shrink semis
    for (BitsetIterator xi(x_merged); xi; ++xi)
        for (BitsetIterator yi(y_merged); yi; ++yi)
        {
            if (*xi == x && *yi == y)
                continue;
            SemiList* merged_semis = m_semis[*xi][*yi];
            if (!merged_semis)
                continue;
            if (!semis)
                semis = m_semis.Put(x, y);
            for (BitsetIterator k(merged_semis->Entries()); k; ++k)
            {
                AndList* key_semis = (*semis)[*k];
                if (!key_semis)
                    key_semis = semis->Put(*k);
                calc_all_semis |= Shrink(added, x, y, key_semis,
                                         *(*merged_semis)[*k], fulls, *k);
            }
            m_semis.Delete(*xi, *yi, merged_semis);
        }

    if (semis)
        for (BitsetIterator k(semis->Entries()); k; ++k)
        {
            AndList* key_semis = (*semis)[*k];
            if (added.test(x) || added.test(y) || newCaptureSet)
            {
                for (CarrierList::Iterator i(*key_semis); i; ++i)
                    if (i.Old())
                        m_semis_and_queue.Push(Semi(x, y, i.Carrier(), *k));
                key_semis->MarkAllUnprocessed();
            }
            else
                key_semis->CalcIntersection();
        }

    // Recalculate all semis if needed
    if (calc_all_semis)
        semis->CalcAllSemis();
    bool anyCpatureSetChanged =
        (oldCapturedSet[x] != m_capturedSet[x]) ||
        (oldCapturedSet[y] != m_capturedSet[y]);
    if (calc_all_semis || (semis && anyCpatureSetChanged))
    {
        if (anyCpatureSetChanged)
            semis->MarkAllUnprocessed();
        if (semis->TryQueue(m_capturedSet[x] | m_capturedSet[y]))
            m_semis_or_queue.Push(Ends(x, y));
    }
}

void VCS::DoSearch()
{
    bool do_threats = m_param->threats;
    while (true)
    {
        if (!m_fulls_and_queue.IsEmpty())
        {
            Full vc = m_fulls_and_queue.Pop();
            AndFull(vc.x, vc.y, vc.carrier);
        }
        else if (!m_semis_and_queue.IsEmpty())
        {
            Semi vc = m_semis_and_queue.Pop();
            AndSemi(vc.x, vc.y, vc.key, vc.carrier);
        }
        else if (!m_semis_or_queue.IsEmpty())
        {
            Ends p = m_semis_or_queue.Pop();
            OrSemis(p.x, p.y);
        }
        else if (do_threats)
        {
            ThreatSearch();
            do_threats = false;
        }
        else
            break;
    }
    TestQueuesEmpty();
    m_fulls_and_queue.Clear();
    m_semis_and_queue.Clear();
    m_semis_or_queue.Clear();;
    TestQueuesEmpty();
}

// And rule stuff

#define FUNC(__name) VCS::VCAnd::Functor##__name()
#define SWITCHTO(__S) return func.template Apply<__S>(*this)

inline VCS::VCAnd::VCAnd(VCS& vcs, HexPoint x, HexPoint y, bitset_t capturedSet,
                         bitset_t xz_carrier, AndList *zy_list, HexPoint key)
    : vcs(vcs), x(x), y(y), capturedSet(capturedSet),
      xz_carrier(xz_carrier), key(key), zy_iter(*zy_list)
{
    BenzeneAssert(x != y);
}

template <class Func>
inline void VCS::VCAnd::Run(Func func)
{
    SWITCHTO(InitialState);
}

template <class S, class Func>
inline void VCS::VCAnd::SemiRemoveSupersetsOf(bitset_t carrier, Func func)
{
    if (!S::semis_initialized)
    {
        semis = vcs.m_semis[x][y];
        if (semis)
            return SemiRemoveSupersetsOf<typename S::SemisSet>(carrier, func);
        else
            return SemiRemoveSupersetsOf<typename S::SemisNull>(carrier, func);
    }
    if (!S::semis_null)
        semis->RemoveSupersetsOf(carrier);
    SWITCHTO(S);
}

template <class S, class Func>
inline void VCS::VCAnd::TryAddFull(bitset_t carrier, Func func)
{
    if (!S::fulls_initialized)
    {
        fulls = vcs.m_fulls[x][y];
        if (fulls)
            return TryAddFull<typename S::FullsSet>(carrier, func);
        else
            return TryAddFull<typename S::FullsNull>(carrier, func);
    }
    if (S::fulls_null)
        fulls = vcs.m_fulls.Put(x, y, new AndList(carrier));
    else if (!fulls->TryAdd(carrier))
        SWITCHTO(S);
    vcs.m_fulls_and_queue.Push(Full(x, y, carrier));
    SemiRemoveSupersetsOf<typename S::FullsSet>(carrier, func);
}

template <class S, class Func>
inline void VCS::VCAnd::TryAddSemi(bitset_t carrier, Func func)
{
    if (!S::semis_initialized)
    {
        semis = vcs.m_semis[x][y];
        if (semis)
            return TryAddSemi<typename S::SemisSet>(carrier, func);
        else
            return TryAddSemi<typename S::SemisNull>(carrier, func);
    }
    if (!S::fulls_initialized)
    {
        fulls = vcs.m_fulls[x][y];
        if (fulls)
            return TryAddSemi<typename S::FullsSet>(carrier, func);
        else
            return TryAddSemi<typename S::FullsNull>(carrier, func);
    }
    if (S::semis_null)
    {
        vcs.m_statistics.and_semi_attempts++;
        if (!S::fulls_null)
        {
            if (fulls->SupersetOfAny(carrier))
                SWITCHTO(S);
        }
        semis = vcs.m_semis.Put(x, y, new SemiList(carrier, key));
        key_semis = (*semis)[key];
    }
    else
    {
        if (!S::key_semis_initialized)
        {
            key_semis = (*semis)[key];
            if (key_semis)
                return TryAddSemi<typename S::KeySemisSet>(carrier, func);
            else
                return TryAddSemi<typename S::KeySemisNull>(carrier, func);
        }
        vcs.m_statistics.and_semi_attempts++;
        if (S::key_semis_null)
        {
            if (!S::fulls_null)
            {
                if (fulls->SupersetOfAny(carrier))
                    SWITCHTO(S);
            }
            key_semis = semis->Put(key, new AndList(carrier));
        }
        else
        {
            if (!S::fulls_null)
            {
                if (fulls->SupersetOfAny(carrier))
                    SWITCHTO(S);
            }
            if (key_semis->SupersetOfAny(carrier))
                SWITCHTO(S);
            key_semis->Add(carrier);
        }
        semis->Add(carrier);
    }

    if (semis->TryQueue(capturedSet))
        vcs.m_semis_or_queue.Push(Ends(x, y));
    if ((vcs.m_brd->GetColor(x) != EMPTY &&
        (vcs.m_param->and_over_edge || !HexPointUtil::isEdge(x)))
        ||
        (vcs.m_brd->GetColor(y) != EMPTY &&
        (vcs.m_param->and_over_edge || !HexPointUtil::isEdge(y))))
        vcs.m_semis_and_queue.Push(Semi(x, y, carrier, key));

    vcs.m_statistics.and_semi_successes++;
    SWITCHTO(typename S::SemisSet::KeySemisSet);
}

template <class S, class Func>
inline void VCS::VCAnd::TryAddSemi(bitset_t carrier, HexPoint new_key,
                                   Func func)
{
    if (new_key != key)
    {
        key = new_key;
        TryAddSemi<typename S::KeySemisReset>(carrier, func);
    }
    else
        TryAddSemi<S>(carrier, func);
}

void VCS::AndFull(HexPoint x, HexPoint y, bitset_t carrier)
{
    BenzeneAssert(x == m_groups->CaptainOf(x));
    BenzeneAssert(y == m_groups->CaptainOf(y));
    BenzeneAssert(x != y);
    BenzeneAssert(m_brd->GetColor(x) != !m_color);
    BenzeneAssert(m_brd->GetColor(y) != !m_color);

    if (!m_fulls[x][y]->TrySetProcessed(carrier))
        return;

    bitset_t xyCapturedSet = m_capturedSet[x] | m_capturedSet[y];
    AndFull(x, y, carrier, m_brd->GetColor(y), xyCapturedSet);
    AndFull(y, x, carrier, m_brd->GetColor(x), xyCapturedSet);
}

inline void VCS::AndFull(HexPoint x, HexPoint z, bitset_t carrier,
                         HexColor zcolor, bitset_t xzCapturedSet)
{
    if (!m_param->and_over_edge && HexPointUtil::isEdge(z))
        return;
    if (zcolor == EMPTY)
        AndFullEmptyFull(x, z, carrier, xzCapturedSet);
    else
    {
        AndFullStoneFull(x, z, carrier, xzCapturedSet);
        AndFullStoneSemi(x, z, carrier, xzCapturedSet);
    }
}

inline void VCS::AndFullEmptyFull(HexPoint x, HexPoint z, bitset_t carrier,
                                  bitset_t xzCapturedSet)
{
    for (BitsetIterator it(m_fulls[z].Entries().reset(x) - carrier); it; ++it)
    {
        HexPoint y = *it;
        BenzeneAssert(y == m_groups->CaptainOf(y));
        BenzeneAssert(x != y && z != y);
        BenzeneAssert(!carrier.test(y));
        AndFullEmptyFull(x, z, y, carrier, xzCapturedSet | m_capturedSet[y]);
    }
}

inline void VCS::AndFullEmptyFull(HexPoint x, HexPoint z, HexPoint y,
                                  bitset_t carrier, bitset_t xyCapturedSet)
{
    AndList* zy_fulls = m_fulls[z][y];
    BenzeneAssert(zy_fulls);
    if (!BitsetUtil::IsSubsetOf(zy_fulls->GetIntersection() & carrier,
        xyCapturedSet))
        return;

    VCAnd vcAnd(*this, x, y, xyCapturedSet, carrier, zy_fulls, z);
    vcAnd.Run(FUNC(FEF));
}

template <class S>
inline void VCS::VCAnd::FEF()
{
    if (!zy_iter)
        return;
    if (!zy_iter.Old())
        return FEFNext<S>();
    if (zy_iter.Carrier().test(x))
        return FEFNext<S>();
    intersection = xz_carrier & zy_iter.Carrier();
    if (intersection.none())
        return TryAddSemi<S>((xz_carrier | zy_iter.Carrier()).set(key), FUNC(FEFNext));
    else if (BitsetUtil::IsSubsetOf(intersection, capturedSet))
        return TryAddSemi<S>((xz_carrier | zy_iter.Carrier() | capturedSet).set(key),
                             FUNC(FEFNext));
    FEFNext<S>();
}

template <class S>
inline void VCS::VCAnd::FEFNext()
{
    ++zy_iter;
    FEF<S>();
}

inline void VCS::AndFullStoneFull(HexPoint x, HexPoint z, bitset_t carrier,
                                  bitset_t xzCapturedSet)
{
    for (BitsetIterator it(m_fulls[z].Entries().reset(x) - carrier); it; ++it)
    {
        HexPoint y = *it;
        BenzeneAssert(y == m_groups->CaptainOf(y));
        BenzeneAssert(x != y && z != y);
        BenzeneAssert(!carrier.test(y));
        AndFullStoneFull(x, z, y, carrier, xzCapturedSet | m_capturedSet[y]);
    }
}

inline void VCS::AndFullStoneFull(HexPoint x, HexPoint z, HexPoint y,
                                  bitset_t carrier, bitset_t xyCapturedSet)
{
    AndList* zy_fulls = m_fulls[z][y];
    BenzeneAssert(zy_fulls);
    if (((zy_fulls->GetIntersection() & carrier) - xyCapturedSet).count() > 1)
        return;

    VCAnd vcAnd(*this, x, y, xyCapturedSet, carrier, zy_fulls);
    vcAnd.Run(FUNC(FSF));
}

template <class S>
inline void VCS::VCAnd::FSF()
{
    if (!zy_iter)
        return;
    if (!zy_iter.Old())
        return FSFNext<S>();
    if (zy_iter.Carrier().test(x))
        return FSFNext<S>();
    intersection = xz_carrier & zy_iter.Carrier();
    BitsetIterator it(intersection);
    if (!it /* intersection is empty */)
        return TryAddFull<S>(xz_carrier | zy_iter.Carrier(), FUNC(FSFNext));
    HexPoint new_key = *it;
    ++it;
    if (!it /* intersection is singleton */)
        return TryAddSemi<S>(xz_carrier | zy_iter.Carrier(), new_key,
                             FUNC(FSFCaptured));
    FSFCaptured<S>();
}

template <class S>
inline void VCS::VCAnd::FSFCaptured()
{
    BitsetIterator it(intersection - capturedSet);
    if (!it /* intersection is empty */)
        return TryAddFull<S>(xz_carrier | zy_iter.Carrier() | capturedSet,
                             FUNC(FSFNext));
    HexPoint new_key = *it;
    ++it;
    if (!it /* intersection is singleton */)
        return TryAddSemi<S>(xz_carrier | zy_iter.Carrier() | capturedSet, new_key,
                             FUNC(FSFNext));
    FSFNext<S>();
}

template <class S>
inline void VCS::VCAnd::FSFNext()
{
    ++zy_iter;
    FSF<S>();
}

inline void VCS::AndFullStoneSemi(HexPoint x, HexPoint z, bitset_t carrier,
                                  bitset_t xzCapturedSet)
{
    for (BitsetIterator it(m_semis[z].Entries().reset(x) - carrier); it; ++it)
    {
        HexPoint y = *it;
        BenzeneAssert(y == m_groups->CaptainOf(y));
        BenzeneAssert(x != y && z != y);
        BenzeneAssert(!carrier.test(y));
        AndFullStoneSemi(x, z, y, carrier, xzCapturedSet | m_capturedSet[y]);
    }
}

inline void VCS::AndFullStoneSemi(HexPoint x, HexPoint z, HexPoint y,
                                  bitset_t carrier, bitset_t xyCapturedSet)
{
    SemiList* zy_semis = m_semis[z][y];
    BenzeneAssert(zy_semis);

    for (BitsetIterator it(zy_semis->Entries()); it; ++it)
    {
        HexPoint key = *it;
        AndFullStoneSemi(x, y, key, carrier, xyCapturedSet,
                         (*zy_semis)[key]);
    }
}

inline void VCS::AndFullStoneSemi(HexPoint x, HexPoint y, HexPoint key,
                                  bitset_t carrier, bitset_t xyCapturedSet,
                                  AndList* zy_list)
{
    BenzeneAssert(zy_list);
    if (!BitsetUtil::IsSubsetOf((zy_list->GetIntersection() & carrier).reset(key),
        xyCapturedSet))
        return;

    VCAnd vcAnd(*this, x, y, xyCapturedSet, carrier, zy_list, key);
    vcAnd.Run(FUNC(FSS));
}


template <class S>
inline void VCS::VCAnd::FSS()
{
    if (!zy_iter)
        return;
    if (!zy_iter.Old())
        return FSSNext<S>();
    if (zy_iter.Carrier().test(x))
        return FSSNext<S>();
    bitset_t intersection = (xz_carrier & zy_iter.Carrier()).reset(key);
    if (intersection.none())
        return TryAddSemi<S>(xz_carrier | zy_iter.Carrier(), FUNC(FSSNext));
    else if (BitsetUtil::IsSubsetOf(intersection, capturedSet))
        return TryAddSemi<S>(xz_carrier | zy_iter.Carrier() | capturedSet, FUNC(FSSNext));
    FSSNext<S>();
}

template <class S>
inline void VCS::VCAnd::FSSNext()
{
    ++zy_iter;
    FSS<S>();
}

void VCS::AndSemi(HexPoint x, HexPoint y, HexPoint key, bitset_t carrier)
{
    BenzeneAssert(x == m_groups->CaptainOf(x));
    BenzeneAssert(y == m_groups->CaptainOf(y));
    BenzeneAssert(x != y);
    BenzeneAssert(m_brd->GetColor(x) != !m_color);
    BenzeneAssert(m_brd->GetColor(y) != !m_color);

    if (!(*m_semis[x][y])[key]->TrySetProcessed(carrier))
        return;

    bitset_t xyCapturedSet = m_capturedSet[x] | m_capturedSet[y];
    AndSemi(x, y, key, carrier, m_brd->GetColor(y), xyCapturedSet);
    AndSemi(y, x, key, carrier, m_brd->GetColor(x), xyCapturedSet);
}

inline void VCS::AndSemi(HexPoint x, HexPoint z, HexPoint key, bitset_t carrier,
                         HexColor zcolor, bitset_t xzCapturedSet)
{
    if (zcolor == EMPTY)
        return;
    if (!m_param->and_over_edge && HexPointUtil::isEdge(z))
        return;
    AndSemiStoneFull(x, z, key, carrier, xzCapturedSet);
}

inline void VCS::AndSemiStoneFull(HexPoint x, HexPoint z, HexPoint key,
                                  bitset_t carrier, bitset_t xzCapturedSet)
{
    for (BitsetIterator it(m_fulls[z].Entries().reset(x) - carrier); it; ++it)
    {
        HexPoint y = *it;
        BenzeneAssert(y == m_groups->CaptainOf(y));
        BenzeneAssert(x != y && z != y);
        BenzeneAssert(!carrier.test(y));
        AndFullStoneSemi(x, y, key, carrier, xzCapturedSet | m_capturedSet[y],
                         m_fulls[z][y]);
    }
}

void VCS::OrSemis(HexPoint x, HexPoint y)
{
    BenzeneAssert(x != y);
    SemiList* xy_semis = m_semis[x][y];
    BenzeneAssert(xy_semis);
    AndList *xy_fulls = m_fulls[x][y];
    m_statistics.doOrs++;
    std::vector<bitset_t> new_fulls =
        VCOr(*xy_semis, xy_fulls ? *xy_fulls : CarrierList(),
             m_capturedSet[x], m_capturedSet[y]);
    xy_semis->MarkAllProcessed();
    if (new_fulls.empty())
        return;
    m_statistics.goodOrs++;
    m_statistics.or_attempts += new_fulls.size();
    m_statistics.or_successes += new_fulls.size();
    if (!xy_fulls)
    {
        for (std::vector<bitset_t>::iterator it = new_fulls.begin();
             it != new_fulls.end(); ++it)
            m_fulls_and_queue.Push(Full(x, y, *it));
        m_fulls.Put(x, y, new AndList(new_fulls));
    }
    else
    {
        for (std::vector<bitset_t>::iterator it = new_fulls.begin();
             it != new_fulls.end(); ++it)
        {
            xy_fulls->Add(*it);
            m_fulls_and_queue.Push(Full(x, y, *it));
        }
    }
}

inline bool VCS::TryAddFull(HexPoint x, HexPoint y, bitset_t carrier)
{
    BenzeneAssert(x != y);
    AndList* fulls = m_fulls[x][y];
    if (!fulls)
        fulls = m_fulls.Put(x, y, new AndList(carrier));
    else if (!fulls->TryAdd(carrier))
        return false;
    m_fulls_and_queue.Push(Full(x, y, carrier));
    SemiList* semis = m_semis[x][y];
    if (semis)
        semis->RemoveSupersetsOf(carrier);
    return true;
}

inline void VCS::TryAddThreat(HexPoint k1, HexPoint k2,
                              AndList* threats, bitset_t carrier)
{
    BenzeneAssert(k1 != k2);
    m_statistics.order2_attempts++;
    if (m_edge_semis)
    {
        if (m_edge_semis->SupersetOfAny(carrier))
            return;
    }
    if (!threats)
        m_threats.Put(k1, k2, new AndList(carrier));
    else if (!threats->TryAdd(carrier))
        return;
    m_statistics.order2_successes++;
}

inline void VCS::ThreatSearch()
{
    HexColorSet not_other = HexColorSetUtil::NotColor(!m_color);
    bitset_t edgesCapturedSet = m_capturedSet[m_edge1] | m_capturedSet[m_edge2];
    m_edge_semis = m_semis[m_edge1][m_edge2];
    for (GroupIterator z(*m_groups, not_other); z; ++z)
    {
        HexPoint zc = z->Captain();
        if (!HexPointUtil::isEdge(zc))
            ThreatSearch(zc, m_brd->GetColor(zc), edgesCapturedSet | m_capturedSet[zc]);
    }

    for (int i = 0; i < BITSETSIZE; i++)
    {
        HexPoint k1 = HexPoint(i);
        const typename BitsetUPairMap<AndList>::Nbs& k1_threats = m_threats[k1];
        if (k1_threats.Entries().none())
            continue;
        AndList threats;
        for (BitsetIterator k(k1_threats.Entries()); k; ++k)
        {
            for (CarrierList::Iterator i(*k1_threats[*k]); i; ++i)
                threats.TryAdd(i.Carrier());
        }

        if (!BitsetUtil::IsSubsetOf(threats.GetIntersection(), edgesCapturedSet))
            continue;
        AndList* k1_semis = m_edge_semis ? (*m_edge_semis)[k1] : 0;
        std::vector<bitset_t> new_semis =
            VCOr(threats, k1_semis ? *k1_semis : CarrierList(),
                 m_capturedSet[m_edge1], m_capturedSet[m_edge2], k1);
        if (new_semis.empty())
            continue;
        if (!k1_semis)
        {
            if (m_edge_semis)
                m_edge_semis = m_semis.Put(m_edge1, m_edge2);
            k1_semis = m_edge_semis->Put(k1);
        }
        for (std::vector<bitset_t>::iterator it = new_semis.begin();
             it != new_semis.end(); ++it)
        {
            k1_semis->Add(*it);
            m_edge_semis->Add(*it);
        }
    }

    m_threats.Reset();

    if (!m_edge_semis)
        return;

    if (m_edge_semis->TryQueue(edgesCapturedSet))
        m_semis_or_queue.Push(Ends(m_edge1, m_edge2));
}

inline void VCS::ThreatSearch(HexPoint z, HexColor zcolor, bitset_t capturedSet)
{
    AndList* fulls1 = m_fulls[m_edge1][z];
    AndList* fulls2 = m_fulls[m_edge2][z];
    SemiList* semis1 = m_semis[m_edge1][z];
    SemiList* semis2 = m_semis[m_edge2][z];
    if (fulls1)
    {
        if (fulls2)
        {
            if (zcolor == EMPTY)
                ThreatSearch1(fulls1, fulls2, capturedSet, z);
            else
                ThreatSearch0(fulls1, fulls2, capturedSet);
        }
        if (semis2)
        {
            if (zcolor == EMPTY)
                ThreatSearch2Semi(fulls1, semis2, capturedSet, z);
            else
                ThreatSearch1Semi(fulls1, semis2, capturedSet);
        }
    }
    if (semis1)
    {
        if (fulls2)
        {
            if (zcolor == EMPTY)
                ThreatSearch2Semi(fulls2, semis1, capturedSet, z);
            else
                ThreatSearch1Semi(fulls2, semis1, capturedSet);
        }
        if (semis2)
        {
            if (zcolor == EMPTY)
                ThreatSearch2SemiSemi(semis1, semis2, capturedSet, z);
            else
                ThreatSearch2SemiSemi(semis1, semis2, capturedSet);
        }
    }
}

inline void VCS::ThreatSearch2SemiSemi(SemiList* semis1, SemiList* semis2,
                                       bitset_t capturedSet, HexPoint k1)
{
    for (BitsetIterator k(semis1->Entries()); k; ++k)
    {
        BenzeneAssert(*k != k1);
        AndList* list2 = (*semis2)[*k];
        if (list2)
            ThreatSearch2((*semis1)[*k], list2, capturedSet, k1, *k);
    }
}

inline void VCS::ThreatSearch2SemiSemi(SemiList* semis1, SemiList* semis2,
                                       bitset_t capturedSet)
{
    for (BitsetIterator k(semis1->Entries()); k; ++k)
        ThreatSearch2Semi((*semis1)[*k], semis2, capturedSet, *k);
}

inline void VCS::ThreatSearch1Semi(AndList* fulls1, SemiList* semis2,
                                   bitset_t capturedSet)
{
    for (BitsetIterator k(semis2->Entries()); k; ++k)
        ThreatSearch1(fulls1, (*semis2)[*k], capturedSet, *k);
}

inline void VCS::ThreatSearch2Semi(AndList* fulls1, SemiList* semis2,
                                   bitset_t capturedSet, HexPoint k1)
{
    for (BitsetIterator k(semis2->Entries()); k; ++k)
        if (k1 == *k)
            ThreatSearch1(fulls1, (*semis2)[*k], capturedSet, k1);
        else
            ThreatSearch2(fulls1, (*semis2)[*k], capturedSet, k1, *k);
}

inline void VCS::ThreatSearch0(AndList* list1, AndList* list2,
                               bitset_t capturedSet)
{
    if (((list1->GetIntersection() & list2->GetIntersection())
        - capturedSet).count() > 2)
        return;
    for (CarrierList::Iterator i(*list1); i; ++i)
        ThreatSearch0(i.Carrier(), list2, capturedSet);
}

inline void VCS::ThreatSearch0(bitset_t carrier1, AndList* list2,
                               bitset_t capturedSet)
{
    if (((carrier1 & list2->GetIntersection())
        - capturedSet).count() > 2)
        return;
    for (CarrierList::Iterator i(*list2); i; ++i)
        ThreatSearch0(carrier1, i.Carrier(), capturedSet);
}

inline void VCS::ThreatSearch0(bitset_t carrier1, bitset_t carrier2,
                               bitset_t capturedSet)
{
    bitset_t intersection = carrier1 & carrier2;
    bitset_t cunion = carrier1 | carrier2;
    if (intersection.count() != 2)
    {
        intersection -= capturedSet;
        if (intersection.count() != 2)
            return;
        cunion |= capturedSet;
    }

    BitsetIterator i(intersection);
    HexPoint k1 = *i;
    ++i;
    HexPoint k2 = *i;
    TryAddThreat(k1, k2, m_threats[k1][k2], cunion);
}

inline void VCS::ThreatSearch1(AndList* list1, AndList* list2,
                               bitset_t capturedSet, HexPoint k1)
{
    if ((((list1->GetIntersection() & list2->GetIntersection())
        - capturedSet)).reset(k1).count() > 1)
        return;
    for (CarrierList::Iterator i(*list1); i; ++i)
        ThreatSearch1(i.Carrier().set(k1), list2, capturedSet, k1);
}

inline void VCS::ThreatSearch1(bitset_t carrier1, AndList* list2,
                               bitset_t capturedSet, HexPoint k1)
{
    if ((((carrier1 & list2->GetIntersection())
        - capturedSet)).reset(k1).count() > 1)
        return;
    for (CarrierList::Iterator i(*list2); i; ++i)
        ThreatSearch1(carrier1, i.Carrier(), capturedSet, k1);
}

inline void VCS::ThreatSearch1(bitset_t carrier1, bitset_t carrier2,
                               bitset_t capturedSet, HexPoint k1)
{
    bitset_t intersection = (carrier1 & carrier2).reset(k1);
    bitset_t cunion = carrier1 | carrier2;
    if (intersection.count() != 1)
    {
        intersection -= capturedSet;
        if (intersection.count() != 1)
            return;
        cunion |= capturedSet;
    }

    BitsetIterator i(intersection);
    HexPoint k2 = *i;
    TryAddThreat(k1, k2, m_threats[k1][k2], cunion);
}

inline void VCS::ThreatSearch2(AndList* list1, AndList* list2,
                               bitset_t capturedSet, HexPoint k1, HexPoint k2)
{
    if ((((list1->GetIntersection() & list2->GetIntersection())
        - capturedSet)).reset(k1).reset(k2).any())
        return;
    for (CarrierList::Iterator i(*list1); i; ++i)
        ThreatSearch2(i.Carrier().set(k1).set(k2), list2, capturedSet, k1, k2);
}

inline void VCS::ThreatSearch2(bitset_t carrier1, AndList* list2,
                               bitset_t capturedSet, HexPoint k1, HexPoint k2)
{
    if ((((carrier1 & list2->GetIntersection())
        - capturedSet)).reset(k1).reset(k2).any())
        return;
    for (CarrierList::Iterator i(*list2); i; ++i)
        ThreatSearch2(carrier1, i.Carrier(), capturedSet, k1, k2);
}

inline void VCS::ThreatSearch2(bitset_t carrier1, bitset_t carrier2,
                               bitset_t capturedSet, HexPoint k1, HexPoint k2)
{
    bitset_t intersection = (carrier1 & carrier2).reset(k1).reset(k2);
    bitset_t cunion = carrier1 | carrier2;
    if (intersection.any())
    {
        intersection -= capturedSet;
        if (intersection.any())
            return;
        cunion |= capturedSet;
    }

    TryAddThreat(k1, k2, m_threats[k1][k2], cunion);
}

inline VCS::Backup::AndListEntry::AndListEntry(HexPoint point, const AndList* andList)
    : point(point),
      andList(new AndList(*andList))
{
}

VCS::Backup::AndListEntry::~AndListEntry()
{
    if (andList)
        delete andList;
}

inline VCS::Backup::FullsEntry::FullsEntry(HexPoint x)
    : x(x)
{
}

inline VCS::Backup::SemiListEntry::SemiListEntry(HexPoint y, const CarrierList& all_semis,
                                                 bitset_t intersection)
    : y(y),
      all_semis(all_semis),
      intersection(intersection)
{
}

inline VCS::Backup::SemisEntry::SemisEntry(HexPoint x)
    : x(x)
{
}

inline void VCS::Backup::Create(VCS& vcs)
{
    vcs.TestQueuesEmpty();
    for (int x = 0; x < BITSETSIZE; x++)
    {
        const typename BitsetUPairMap<AndList>::Nbs& nbs = vcs.m_fulls[(HexPoint)x];
        if (nbs.Entries().none())
            continue;
        FullsEntry entry((HexPoint)x);
        for (BitsetIterator y(nbs.Entries()); y; ++y)
            entry.list.push_back(AndListEntry(*y, nbs[*y]));
        fulls.push_back(entry);
    }
    for (int x = 0; x < BITSETSIZE; x++)
    {
        const typename BitsetUPairMap<SemiList>::Nbs& nbs = vcs.m_semis[(HexPoint)x];
        if (nbs.Entries().none())
            continue;
        SemisEntry entry((HexPoint)x);
        for (BitsetIterator y(nbs.Entries()); y; ++y)
        {
            SemiList* semi_list = nbs[*y];
            SemiListEntry keys(*y, *semi_list, semi_list->GetIntersection());
            for (BitsetIterator key(semi_list->Entries()); key; ++key)
                keys.list.push_back(AndListEntry(*key, (*semi_list)[*key]));
            entry.list.push_back(keys);
        }
        semis.push_back(entry);
    }
}

inline void VCS::Backup::Restore(VCS& vcs)
{
    vcs.TestQueuesEmpty();
    for (std::vector<FullsEntry>::iterator entry = fulls.begin();
         entry != fulls.end(); ++entry)
    {
        for (std::vector<AndListEntry>::iterator i = entry->list.begin();
             i != entry->list.end(); ++i)
        {
            vcs.m_fulls.Put(entry->x, i->point, i->andList);
            i->andList = 0;
        }
    }
    for (std::vector<SemisEntry>::iterator entry = semis.begin();
         entry != semis.end(); ++entry)
    {
        for (std::vector<SemiListEntry>::iterator i = entry->list.begin();
            i != entry->list.end(); ++i)
        {
            SemiList* semi_list = new SemiList(i->all_semis, i->intersection);
            for (std::vector<AndListEntry>::iterator k = i->list.begin();
                 k != i->list.end(); ++k)
            {
                semi_list->Put(k->point, k->andList);
                k->andList = 0;
            }
            vcs.m_semis.Put(entry->x, i->y, semi_list);
        }
    }
}

bitset_t VCS::GetSmallestSemisUnion() const
{
    return SemiGreedyUnion(m_edge1, m_edge2);
}

bool VCS::SmallestFullCarrier(bitset_t& carrier) const
{
    AndList* fulls = m_fulls[m_edge1][m_edge2];
    if (!fulls)
        return false;
    if (fulls->IsEmpty())
        return false;
    size_t best = std::numeric_limits<size_t>::max();
    for (CarrierList::Iterator i(*fulls); i; ++i)
    {
        size_t count = i.Carrier().count();
        if (count < best)
        {
            best = count;
            carrier = i.Carrier();
        }
    }
    return true;
}

int VCS::FullAdjacent(HexPoint x, HexPoint y) const
{
    AndList* fulls = m_fulls[x][y];
    if (!fulls)
        return -1;
    if (fulls->IsEmpty())
        return -1;
    CarrierList::Iterator i(*fulls);
    return i.Carrier().any() ? 1 : 0;
}

bool VCS::SmallestSemiCarrier(bitset_t& carrier) const
{
    SemiList* semis = m_semis[m_edge1][m_edge2];
    if (!semis)
        return false;
    bool res = false;
    size_t best = std::numeric_limits<size_t>::max();
    for (CarrierList::Iterator i(*semis); i; ++i)
    {
        size_t count = i.Carrier().count();
        if (count < best)
        {
            best = count;
            carrier = i.Carrier();
        }
        res = true;
    }
    return res;
}

HexPoint VCS::SmallestSemiKey() const
{
    SemiList* semis = m_semis[m_edge1][m_edge2];
    if (!semis)
        return INVALID_POINT;
    size_t best = std::numeric_limits<size_t>::max();
    HexPoint res = INVALID_POINT;
    for (BitsetIterator it(semis->Entries()); it; ++it)
    {
        HexPoint key = *it;
        for (CarrierList::Iterator i(*(*semis)[key]); i; ++i)
        {
            size_t count = i.Carrier().count();
            if (count < best)
            {
                best = count;
                res = key;
            }
        }
    }
    return res;
}

bool VCS::FullExists(HexPoint x, HexPoint y) const
{
    AndList* fulls = m_fulls[x][y];
    if (!fulls)
        return false;
    return !fulls->IsEmpty();
}

bool VCS::FullExists() const
{
    return FullExists(m_edge1, m_edge2);
}

bool VCS::SemiExists() const
{
    SemiList* semis = m_semis[m_edge1][m_edge2];
    if (!semis)
        return false;
    return !semis->IsEmpty();
}

static CarrierList empty_carrier_list;

const CarrierList& VCS::GetFullCarriers(HexPoint x, HexPoint y) const
{
    AndList* fulls = m_fulls[x][y];
    if (!fulls)
        return empty_carrier_list;
    return *fulls;
}

const CarrierList& VCS::GetFullCarriers() const
{
    return GetFullCarriers(m_edge1, m_edge2);
}

const CarrierList& VCS::GetSemiCarriers() const
{
    SemiList* semis = m_semis[m_edge1][m_edge2];
    if (!semis)
        return empty_carrier_list;
    return *semis;
}

bitset_t VCS::GetFullNbs(HexPoint x) const
{
    return m_fulls[x].Entries();
}

bitset_t VCS::GetSemiNbs(HexPoint x) const
{
    return m_semis[x].Entries();
}

bitset_t VCS::FullIntersection(HexPoint x, HexPoint y) const
{
    AndList* fulls = m_fulls[x][y];
    return fulls ? fulls->GetAllIntersection() : bitset_t().set();
}

bitset_t VCS::FullGreedyUnion(HexPoint x, HexPoint y) const
{
    AndList* fulls = m_fulls[x][y];
    return fulls ? fulls->GetGreedyUnion() : EMPTY_BITSET;
}

bitset_t VCS::SemiIntersection(HexPoint x, HexPoint y) const
{
    SemiList* semis = m_semis[x][y];
    return semis ? semis->GetIntersection() : bitset_t().set();
}

bitset_t VCS::SemiIntersection() const
{
    return SemiIntersection(m_edge1, m_edge2);
}

bitset_t VCS::SemiGreedyUnion(HexPoint x, HexPoint y) const
{
    SemiList* semis = m_semis[x][y];
    return semis ? semis->GetGreedyUnion() : EMPTY_BITSET;
}
