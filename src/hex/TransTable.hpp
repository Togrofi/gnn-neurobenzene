//----------------------------------------------------------------------------
/** @file TransTable.hpp
 */
//----------------------------------------------------------------------------

#ifndef TRANSTABLE_H
#define TRANSTABLE_H

#include <boost/concept_check.hpp>

#include "Hex.hpp"

_BEGIN_BENZENE_NAMESPACE_

//----------------------------------------------------------------------------

/** Concept of a state in a transposition table. */
template<class T>
struct TransTableStateConcept
{
    void constraints() 
    {
        boost::function_requires< boost::DefaultConstructibleConcept<T> >();
        boost::function_requires< boost::AssignableConcept<T> >();

        const T a;
        if (a.Initialized()) { }

        const T b;
        if (a.ReplaceWith(b)) { }
    }
};

//----------------------------------------------------------------------------

/** Transposition table. The state class must meet the requirements of
    TransTableStateConcept.  */
template<class T>
class TransTable
{
    BOOST_CLASS_REQUIRE(T, benzene, TransTableStateConcept);

public:

    /** Creates table with 2^n slots. */
    TransTable(int bits);

    /** Destructor. */
    ~TransTable();

    /** Returns lg2 of number of entries. */
    std::size_t Bits() const;

    /** Returns the number of slots in the TT. */
    std::size_t Size() const;

    /** Clears the table. */
    void Clear();

    /** Stores data in slot for hash. New data
        overwrites old only if "old.ReplaceWith(new)" is true.
    */
    bool Put(hash_t hash, const T& data);
    
    /** Returns true if the slot for hash contains a state with that
        hash value, data is copied into data if so. Otherwise, nothing
        is copied into data.
    */
    bool Get(hash_t hash, T& data);

    /** Returns statistics in string form. */
    std::string Stats() const;

private:

    // -----------------------------------------------------------------------

    struct Statistics
    {
        unsigned reads;
        unsigned hits;
        unsigned writes;

        Statistics() : reads(0), hits(0), writes(0) { }
    };
   
    // -----------------------------------------------------------------------
    
    int m_bits;

    std::size_t m_size;

    std::size_t m_mask;

    std::vector<T> m_data;

    std::vector<hash_t> m_hash;

    Statistics m_stats;
};

//----------------------------------------------------------------------------

template<typename T>
TransTable<T>::TransTable(int bits)
    : m_bits(bits),
      m_size(1 << bits), 
      m_mask(m_size - 1),
      m_data(m_size),
      m_hash(m_size, 0),
      m_stats()
{
    Clear();
}

template<typename T>
TransTable<T>::~TransTable()
{
}

template<typename T>
inline std::size_t TransTable<T>::Bits() const
{
    return m_bits;
}

template<typename T>
inline std::size_t TransTable<T>::Size() const
{
    return m_size;
}

template<typename T>
inline void TransTable<T>::Clear()
{
    for (std::size_t i = 0; i < m_size; ++i)
    {
        m_data[i] = T();
        m_hash[i] = 0;
    }
}

template<typename T>
bool TransTable<T>::Put(hash_t hash, const T& data)
{
    std::size_t slot = hash & m_mask;
    if (m_data[slot].ReplaceWith(data)) 
    {
        m_stats.writes++;
        m_data[slot] = data;
        m_hash[slot] = hash;
    }
    return true;
}

template<typename T>
bool TransTable<T>::Get(hash_t hash, T& data)
{
    m_stats.reads++;
    std::size_t slot = hash & m_mask;
    T& old = m_data[slot];
    bool ret = old.Initialized() && (m_hash[slot] == hash);
    if (ret) 
    {
        data = old;
        m_stats.hits++; 
    }
    return ret;
}

template<typename T>
std::string TransTable<T>::Stats() const
{  
    std::ostringstream os;
    os << '\n'
       << "TT statistics\n"
       << "   reads: " << m_stats.reads << std::endl
       << "    hits: " << m_stats.hits << std::endl
       << "  writes: " << m_stats.writes << std::endl;
    return os.str();
}

//----------------------------------------------------------------------------

_END_BENZENE_NAMESPACE_

#endif // TRANSTABLE_H
