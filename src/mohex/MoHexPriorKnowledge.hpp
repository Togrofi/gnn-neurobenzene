//----------------------------------------------------------------------------
/** @file MoHexPriorKnowledge.hpp */
//----------------------------------------------------------------------------

#ifndef MOHEXPRIORKNOWLEDGE_HPP
#define MOHEXPRIORKNOWLEDGE_HPP

_BEGIN_BENZENE_NAMESPACE_

//----------------------------------------------------------------------------

class MoHexThreadState;

/** Applies knowledge to set of moves. */
class MoHexPriorKnowledge
{
public:
    MoHexPriorKnowledge(const MoHexThreadState& m_state);
    //MoHexPriorKnowledge(MoHexThreadState& m_state); //is this okay to remove const?


    ~MoHexPriorKnowledge();

    SgUctValue ProcessPosition(std::vector<SgUctMoveInfo>& moves,
                         const HexPoint lastMove, const bool doPruning);
    
private:
    const MoHexThreadState& m_state;
    //is it okay to remove the const?
    //MoHexThreadState& m_state;

};

//----------------------------------------------------------------------------

_END_BENZENE_NAMESPACE_

#endif // MOHEXPRIORKNOWLEDGE_HPP
