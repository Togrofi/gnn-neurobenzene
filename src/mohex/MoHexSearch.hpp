//----------------------------------------------------------------------------
/** @file MoHexSearch.hpp */
//----------------------------------------------------------------------------

#ifndef MOHEXSEARCH_H
#define MOHEXSEARCH_H

#include "SgBlackWhite.h"
#include "SgPoint.h"
#include "SgNode.h"
#include "SgUctSearch.h"

#include "MoHexThreadState.hpp"
#include "MoHexPatterns.hpp"
#include "NeuroEvaluate.hpp"


_BEGIN_BENZENE_NAMESPACE_

//----------------------------------------------------------------------------

class MoHexSharedPolicy;

/** Creates threads. */
class HexThreadStateFactory : public SgUctThreadStateFactory
{
public:
    HexThreadStateFactory(MoHexSharedPolicy* shared_policy);

    ~HexThreadStateFactory();

    SgUctThreadState* Create(unsigned int threadId, const SgUctSearch& search);
private:

    MoHexSharedPolicy* m_shared_policy;
};

//----------------------------------------------------------------------------

/** Monte-Carlo search using UCT for Hex. */
class MoHexSearch : public SgUctSearch
{
public:
    /** Constructor.
        @param factory Creates MoHexState instances for each thread.
        @param maxMoves Maximum move number.
    */
    MoHexSearch(SgUctThreadStateFactory* factory,
                int maxMoves = 0);
    
    ~MoHexSearch();    

    //-----------------------------------------------------------------------
    
    /** @name Pure virtual functions of SgUctSearch */
    // @{

    std::string MoveString(SgMove move) const;

    SgUctValue UnknownEval() const;

    SgUctValue InverseEval(SgUctValue eval) const;

    // @}

    //-----------------------------------------------------------------------

    /** @name Virtual functions of SgUctSearch */
    // @{

    void OnSearchIteration(SgUctValue gameNumber, const unsigned int threadId,
                           const SgUctGameInfo& info);

    void OnStartSearch();

    // @}

    //-----------------------------------------------------------------------

    /** @name Hex-specific functions */
    // @{

    void SetBoard(HexBoard& board);

    HexBoard& Board();

    const HexBoard& Board() const;

    void SetSharedData(MoHexSharedData& data);

    MoHexSharedData& SharedData();

    const MoHexSharedData& SharedData() const;

    const MoHexPatterns& GlobalPatterns() const;

    const MoHexPatterns& LocalPatterns() const;

    const MoHexPatterns& PlayoutGlobalPatterns() const;

    const MoHexPatterns& PlayoutLocalPatterns() const;

    void LoadPatterns();

    /** @see MoHexUtil::SaveTree() */
    void SaveTree(std::ostream& out, int maxDepth) const;

    // @}

    //-----------------------------------------------------------------------

    /** @name Hex-specific parameters */
    // @{

    /** Enables output of live graphics commands for HexGui.
        See GoGuiGfx() */
    void SetLiveGfx(bool enable);

    /** See SetLiveGfx(). */
    bool LiveGfx() const;

    /** Size of the map of fillin states. */
    int FillinMapBits() const;

    /** See FillinMapBits(). */
    void SetFillinMapBits(int bits);

    /** Whether to prune moves during prior computation. */
    bool PriorPruning() const;

    /** See PriorPruning() */
    void SetPriorPruning(bool enable);

    /** Gamma for VC maintenance moves. */
    float VCMGamma() const;

    void SetVCMGamma(float gamma);

    /** added by Chao Gao */
    const NNEvaluator& GetNNEvaluator() const;

	std::shared_ptr<NNEvaluator> GetNNEvaluatorPtr() const;

    void SetNNEvaluator(std::shared_ptr<NNEvaluator> nn_ptr);

    float RootDirichletPrior() const;

    void SetRootDirichletPrior(float value);
    // @} 

private:
    /** See SetKeepGames() */
    bool m_keepGames;

    /** See SetLiveGfx() */
    bool m_liveGfx;

    /** Nothing is done to this board. 
        We do not own this pointer. Threads will create their own
        HexBoards, but the settings (ICE and VCs) will be copied from
        this board. */
    HexBoard* m_brd;

    int m_fillinMapBits;

    /** See PriorPruning() */
    bool m_priorPruning;
   
    float m_vcmGamma;

    /** Data among threads. */
    boost::scoped_ptr<MoHexSharedData> m_sharedData;

    StoneBoard m_lastPositionSearched;

    SgUctValue m_nextLiveGfx;

    MoHexPatterns m_globalPatterns;

    MoHexPatterns m_localPatterns;

    MoHexPatterns m_playoutGlobalPatterns;

    MoHexPatterns m_playoutLocalPatterns;

    /** Not implemented */
    MoHexSearch(const MoHexSearch& search);

    /** Not implemented */
    MoHexSearch& operator=(const MoHexSearch& search);

    /** added by Chao Gao */
    std::shared_ptr<NNEvaluator> m_nnEvaluator;

    /** dirichlet nosie at root; 0.0 means no dirichlet noise otherwise
     * sameple from dirichlet distribution for combine root prior */
    float m_root_dirichlet_prior;
};

inline float MoHexSearch::RootDirichletPrior() const {
    return m_root_dirichlet_prior;
}

inline void MoHexSearch::SetRootDirichletPrior(float value){
    m_root_dirichlet_prior=value;
}

inline const NNEvaluator& MoHexSearch::GetNNEvaluator() const {
    return *m_nnEvaluator;
}

inline std::shared_ptr<NNEvaluator> MoHexSearch::GetNNEvaluatorPtr() const
{
	return m_nnEvaluator;
}

inline void MoHexSearch::SetNNEvaluator(std::shared_ptr<NNEvaluator> nn_ptr){
    m_nnEvaluator=nn_ptr;
}

inline void MoHexSearch::SetBoard(HexBoard& board)
{
    m_brd = &board;
}

inline HexBoard& MoHexSearch::Board()
{
    return *m_brd;
}

inline const HexBoard& MoHexSearch::Board() const
{
    return *m_brd;
}

inline bool MoHexSearch::LiveGfx() const
{
    return m_liveGfx;
}

inline void MoHexSearch::SetLiveGfx(bool enable)
{
    m_liveGfx = enable;
}

inline void MoHexSearch::SetSharedData(MoHexSharedData& data)
{
    m_sharedData.reset(new MoHexSharedData(data));
}

inline MoHexSharedData& MoHexSearch::SharedData()
{
    return *m_sharedData;
}

inline const MoHexSharedData& MoHexSearch::SharedData() const
{
    return *m_sharedData;
}

inline int MoHexSearch::FillinMapBits() const
{
    return m_fillinMapBits;
}

inline void MoHexSearch::SetFillinMapBits(int bits)
{
    m_fillinMapBits = bits;
}

inline const MoHexPatterns& MoHexSearch::GlobalPatterns() const
{
    return m_globalPatterns;
}

inline const MoHexPatterns& MoHexSearch::LocalPatterns() const
{
    return m_localPatterns;
}

inline const MoHexPatterns& MoHexSearch::PlayoutGlobalPatterns() const
{
    return m_playoutGlobalPatterns;
}

inline const MoHexPatterns& MoHexSearch::PlayoutLocalPatterns() const
{
    return m_playoutLocalPatterns;
}

inline bool MoHexSearch::PriorPruning() const
{
    return m_priorPruning;
}

inline void MoHexSearch::SetPriorPruning(bool enable)
{
    m_priorPruning = enable;
}

inline float MoHexSearch::VCMGamma() const
{
    return m_vcmGamma;
}

inline void MoHexSearch::SetVCMGamma(float gamma)
{
    m_vcmGamma = gamma;
}

//----------------------------------------------------------------------------

_END_BENZENE_NAMESPACE_

#endif // MOHEXSEARCH_H
