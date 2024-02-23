#include<iostream>
#include<vector>
#include<utility>
#include <ConstBoard.hpp>
#include <cfloat>
#include <cassert>

#include "NeuroEvaluate.hpp"

typedef std::vector<float> v1d;
typedef std::vector<v1d> v2d;
typedef std::vector<v2d> v3d;
typedef std::vector<v3d> v4d;
typedef torch::Tensor Tensor;

NNEvaluator::NNEvaluator(){
}

NNEvaluator::NNEvaluator(std::string model_path):m_module(),
    m_min_q_combine_weight(0.0), m_q_weight_to_p(0.0), m_product_propagate_weight(0.0) {
    load_nn_model(model_path);
}

void NNEvaluator::load_nn_model(std::string model_path){
    this->m_module = torch::jit::load(model_path, torch::kCPU);
}
Tensor* NNEvaluator::make_input_tensor(const benzene::bitset_t &black_stones,
                                    const benzene::bitset_t &white_stones, benzene::HexColor toplay, 
                                    int boardsize) const {
    size_t i,j, k;
    int x,y;
    const std::vector<std::int64_t> input_dims = {1, boardsize*boardsize, m_input_depth};
    float input_vals[1][boardsize*boardsize][m_input_depth];
    //set the empty points
    for (i=0;i<boardsize;i++){
        for(j=0;j<boardsize;j++){
            for(k=0;k<m_input_depth; k++){
                input_vals[0][i][j][k]=0;
                if(k==ToPlayEmptyPoints && i>=m_input_padding && j>=m_input_padding
                        && i<boardsize-m_input_padding && j<boardsize-m_input_padding){
                    input_vals[0][i][j][k]=1.0;
                    x=i-m_input_padding;
                    y=j-m_input_padding;
                    //static_assert(x>=0 && x<boardsize && y>=0 && y<boardsize, "error in make_input_tensor\n");
                    //be aware of this conversion! pari to int_move: x*boarsize + y

                } 
                //toplay plane
                if(k==IndToPlay && toplay==benzene::WHITE){
                    input_vals[0][i][j][k]=1.0;
                }
            }
        }
    }

    //set m_board, and black/white played stones, modify empty points
    for (benzene::BitsetIterator it(black_stones); it; ++it){
        int p=*it-7;
        if (p<0) continue;
        benzene::HexPointUtil::pointToCoords(*it, x,y);
        x += m_input_padding;
        y += m_input_padding;
        input_vals[0][x][y][BlackStones]=1.0;
        input_vals[0][x][y][ToPlayEmptyPoints]=0.0;
    }
    for (benzene::BitsetIterator it(white_stones); it; ++it){
        int p=*it-7;
        if(p<0) continue;
        benzene::HexPointUtil::pointToCoords(*it,x,y);
        x += m_input_padding;
        y += m_input_padding;
        input_vals[0][x][y][WhiteStones]=1.0;
        input_vals[0][x][y][ToPlayEmptyPoints]=0.0;
    }
    float *p_input_vals=(float*)input_vals;
    Tensor* input_tensor = ;

    return input_tensor;
}

/*
 * return:
 * p: probability score of next moves
 * q: action-value for each next move
 * v: value estimate of current state
 */
float NNEvaluator::evaluate(const benzene::bitset_t &black, const benzene::bitset_t &white, benzene::HexColor toplay,
                            std::vector<float> &score, std::vector<float> & qValues, int boardsize) const {
    auto t1=std::chrono::system_clock::now();
    Tensor* x_input=make_input_tensor(black, white, toplay, boardsize);
    const std::vector<Tensor*> input_tensors={x_input};

    int output = m_module();

    float* p_ret=;
    float* v_ret=;
    float* q_ret=;
    float value_estimate=v_ret[0];

    float max_value=-FLT_MAX;
    std::vector<int> empty_points;
    int max_ind=-1;
    float min_q_value=1.0;
    for(int i=0;i<boardsize*boardsize;i++){
        int x, y;
        x= i/boardsize;
        y= i%boardsize;
        size_t posi=benzene::HexPointUtil::coordsToPoint(x,y);
        if(black.test(posi) || white.test(posi)){
            //ignore played points
            score[i]=0.0;
            continue;
        }
        score[i]=0.0;
        empty_points.push_back(i);
        if(max_value < p_ret[i]){
            max_value=p_ret[i];
            max_ind=i;
        }
        if(min_q_value>q_ret[i]){
            min_q_value=q_ret[i];
        }
        qValues[i]=(q_ret[i]+1.0)/2.0f;//convert to [0,1]
        score[i]=p_ret[i];
    }
    double product_propagate_prob=1.0;
    float sum_value=0.0;
    for(int &i:empty_points){
        score[i]=(float)exp((score[i]-max_value));
        sum_value = sum_value+score[i];
    }
    double avg_v=0.0;
    double q_value_with_max_p;
    double max_p=-1.0;
    for(int &i: empty_points){
        score[i]=score[i]/sum_value;
        avg_v += score[i]*(1.0-qValues[i]);
        score[i]=(1.0-m_q_weight_to_p)*score[i]+m_q_weight_to_p*(1.0-qValues[i]);
        if(score[i]>=0.01) 
            product_propagate_prob *= qValues[i];
        if(score[i]>max_p){
            max_p=score[i];
            q_value_with_max_p=(1.0-qValues[i]);
        }
    }
    //convert from [-1.0,1.0] to [0.0,1.0]
    double converted_min_q_value=(-min_q_value+1.0)/2.0f;
    converted_min_q_value=q_value_with_max_p;
    double converted_v_value=(1.0+value_estimate)/2.0f;
    value_estimate=(1.0-m_min_q_combine_weight)*converted_v_value+m_min_q_combine_weight*converted_min_q_value;
    value_estimate=(1.0-m_product_propagate_weight)*value_estimate + m_product_propagate_weight*(1.0-product_propagate_prob);

    auto t2=std::chrono::system_clock::now();
    //std::cout<<"time cost per eva:"<<std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()/1e06<<" seconds\n";
    return value_estimate;
}
