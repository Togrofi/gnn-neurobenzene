#include<iostream>
#include <torch/script.h> // One-stop header.
#include <stdexcept>
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
    this->m_min_q_combine_weight = 0.0;
    this->m_q_weight_to_p = 0.0;
    this->m_product_propagate_weight = 0.0f;
    load_nn_model(model_path);
}

void NNEvaluator::load_nn_model(std::string model_path){
    this->m_module = torch::jit::load(model_path, torch::kCPU);
}
std::vector<torch::jit::IValue>
NNEvaluator::make_input_tensor(const benzene::bitset_t &black_stones,
                                    const benzene::bitset_t &white_stones, benzene::HexColor toplay, 
                                    int boardsize) const {

    Tensor board_state = torch::zeros({1, boardsize*boardsize, m_input_depth});
    Tensor adj_matrix = torch::zeros({boardsize*boardsize, boardsize*boardsize});

    auto is_coord_in_board = [](int x, int y, int boardsize) {
        return x >= 0 && x < boardsize && y >= 0 && y < boardsize;
    };

    auto index_from_position = [=](int row, int col, int boardsize) {
        return row * boardsize + col;
    };

    for (int i = 0; i < boardsize; ++i) {
        for (int j = 0; j < boardsize; ++j) {
            std::vector<std::pair<int, int>> potential_neighbours = {
                {i, j + 1}, {i + 1, j}, {i - 1, j}, {i, j - 1}, {i - 1, j + 1}, {i + 1, j - 1}
            };
            for (const auto& coord : potential_neighbours) {
                int x = coord.first;
                int y = coord.second;
                if (is_coord_in_board(x, y, boardsize)) {
                    adj_matrix[index_from_position(i, j, boardsize)][index_from_position(x, y, boardsize)] = 1;
                }
            }
        }

    for (int i = 0; i<boardsize*boardsize; i++) {
       board_state[0][i][ToPlayEmptyPoints] = 1.0;
    }

    //set m_board, and black/white played stones, modify empty points
    for (benzene::BitsetIterator it(black_stones); it; ++it){
        int p=*it-7;
        if (p<0) continue;

        if (toplay==benzene::WHITE) {
            board_state[0][p][BlackStones]=1.0;
        } else {
            board_state[0][p][WhiteStones]=1.0;
        }
        board_state[0][p][ToPlayEmptyPoints]=0.0;
    }

    for (benzene::BitsetIterator it(white_stones); it; ++it){
        int p=*it-7;
        if(p<0) continue;

        if (toplay==benzene::WHITE) {
            board_state[0][p][WhiteStones]=1.0;
        } else {
            board_state[0][p][BlackStones]=1.0;
        }
        board_state[0][p][ToPlayEmptyPoints]=0.0;
    }

    std::vector<torch::jit::IValue> input_tensor;
    input_tensor.push_back(board_state);
    input_tensor.push_back(adj_matrix);
    return input_tensor;
}

/*
 * return:
 * p: probability prob_score of next moves
 * q: action-value for each next move
 * v: value estimate of current state
 */
float NNEvaluator::evaluate(const benzene::bitset_t &black, const benzene::bitset_t &white, benzene::HexColor toplay,
                            std::vector<float> &prob_score, std::vector<float> & qValues, int boardsize) const {
    auto t1=std::chrono::system_clock::now();
    std::vector<torch::jit::IValue> inputs=make_input_tensor(black, white, toplay, boardsize);

    torch::Tensor output = m_module.forward(inputs).toTensor();
    std::cout << output.sizes() << '\n';
    std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
    float* p_ret;
    float* v_ret;
    float* q_ret;
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
            prob_score[i]=0.0;
            continue;
        }
        prob_score[i]=0.0;
        empty_points.push_back(i);
        if(max_value < p_ret[i]){
            max_value=p_ret[i];
            max_ind=i;
        }
        if(min_q_value>q_ret[i]){
            min_q_value=q_ret[i];
        }
        qValues[i]=(q_ret[i]+1.0)/2.0f;//convert to [0,1]
        prob_score[i]=p_ret[i];
    }
    double product_propagate_prob=1.0;
    float sum_value=0.0;
    for(int &i:empty_points){
        prob_score[i]=(float)exp((prob_score[i]-max_value));
        sum_value = sum_value+prob_score[i];
    }
    double avg_v=0.0;
    double q_value_with_max_p;
    double max_p=-1.0;
    for(int &i: empty_points){
        prob_score[i]=prob_score[i]/sum_value;
        avg_v += prob_score[i]*(1.0-qValues[i]);
        prob_score[i]=(1.0-m_q_weight_to_p)*prob_score[i]+m_q_weight_to_p*(1.0-qValues[i]);
        if(prob_score[i]>=0.01)
            product_propagate_prob *= qValues[i];
        if(prob_score[i]>max_p){
            max_p=prob_score[i];
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
