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





class TEncDecisionTree{
private:
    static double calcAverageDepthIterative(TComDataCU *&cu);
    static double getFeatureValue(TComDataCU *&cu, int feat_idx);
    static void readTree();
    static int classifyCU(string dataRec, int depth);
    
public:
    static bool enabled;
    static int encodedFrames,nonZeroCoeff, deltaQP;
    static double cost_2Nx2N, cost_MSM, SAD, SSE, neighDepth;
    static string dataRec;

    static void init( );


    
    static void getSADSSE( TComYuv *pcYuvSrc0, TComYuv *pcYuvSrc1,int width);
    static double getAverageNeighborDepth(TComDataCU *&cu);
    static string setCUFeatures(TComDataCU *&cu);
    static bool decideSplit(TComDataCU *&cu, int depth);
};

#endif	/* TENCDECISIONTREE_H */

