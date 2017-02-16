/* 
 * File:   TEncDecisionTree.h
 * Author: grellert
 *
 * Created on February 9, 2017, 3:38 PM
 */

#ifndef TENCDECISIONTREE_H
#define	TENCDECISIONTREE_H

#include "../TLibCommon/TComDataCU.h"
#include "../TLibCommon/TComPic.h"
#include "../TLibCommon/TComYuv.h"
#include<vector>
#include<map>


#define ONLINE_TRAIN 1
#define NB_TRAINING_FRAMES 4
#define NB_FEATURES 28
#define WRITE_TEST 0

class TEncDecisionTree{
private:
    static double calcAverageDepthIterative(TComDataCU *&cu);
    static double getFeatureValue(TComDataCU *&cu, int feat_idx);
    static void readTree();
    static int classifyCU(string dataRec, int depth);
    static void runC50Train();
public:

    static bool enabled, boosting, trained, encodeStarted;
    static int encodedFrames,nonZeroCoeff, deltaQP, QP;
    static double cost_2Nx2N, cost_MSM, SAD, SSE, neighDepth;
    static std::string dataRec;
    static std::string sequence;
    static FILE *trainFile[3],*testFile[3];
    
    static std::vector<std::string > cuOrderMap; 
    static std::map<std::string, std::string > statsMap; 


    static void init( );

    static void getSADSSE( TComYuv *pcYuvSrc0, TComYuv *pcYuvSrc1,int width);
    static double getAverageNeighborDepth(TComDataCU *&cu);
    static string setCUFeatures(TComDataCU *&cu);
    static bool decideSplit(TComDataCU *&cu, int depth);
    static bool trainingPhase();
    static bool predictPhase();
    static std::string getMapString(TComDataCU *cu, int depth, int partIdx);
    static void writeTrainFile();

};

#endif	/* TENCDECISIONTREE_H */

