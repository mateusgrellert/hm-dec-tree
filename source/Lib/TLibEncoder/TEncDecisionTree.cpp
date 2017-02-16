#include "TEncDecisionTree.h"
#include <math.h>
#include <sstream>

#ifndef C50
#define	C50
#include "../C50/defns.h"
#include "../C50/global.c"
#include "../C50/hooks.c"
#endif

#include<algorithm>
double TEncDecisionTree::cost_2Nx2N;
double TEncDecisionTree::cost_MSM;
double TEncDecisionTree::neighDepth;
int TEncDecisionTree::nonZeroCoeff;
int TEncDecisionTree::deltaQP;
int TEncDecisionTree::QP;
int TEncDecisionTree::encodedFrames;
bool TEncDecisionTree::enabled;
bool TEncDecisionTree::boosting;
bool TEncDecisionTree::trained;
bool TEncDecisionTree::encodeStarted;
double TEncDecisionTree::SAD;
double TEncDecisionTree::SSE;
std::string TEncDecisionTree::sequence;
std::string TEncDecisionTree::trainPath[3];
std::string TEncDecisionTree::outPath[3];
std::string TEncDecisionTree::testPath[3];
string TEncDecisionTree::dataRec;
FILE* TEncDecisionTree::trainFile[3];
FILE* TEncDecisionTree::testFile[3];

std::map<std::string, std::string > TEncDecisionTree::statsMap; 
std::vector<std::string > TEncDecisionTree::cuOrderMap; 

void TEncDecisionTree::init(){
    int d;

    dataRec = string();
    
#if !ONLINE_TRAIN
    
    for (d = 0 ; d <= 2; d++){
        std::stringstream sstr_tr;
        
        sstr_tr << "./OFFLINE/";
        if(boosting) sstr_tr << "BOOST/";
        else         sstr_tr << "NO-BOOST/";
        
        sstr_tr << "TRA_NEB_BDrv_PkS_PtS_BQM_BBub_RH_d" << d;
        trainPath[d] = sstr_tr.str();
        
        gDepth = d;
        readTree();
    }
#endif
    trained = false;
    encodeStarted = false;
    
    
    for(d = 0; d <= 2; d++){
 
#if ONLINE_TRAIN
        std::stringstream sstr_tr,sstr_out;

        sstr_tr <<  "./ONLINE/";
        sstr_out << "./ONLINE/";
        
        if(boosting) sstr_tr <<  "BOOST/";
        else         sstr_tr <<  "NO-BOOST/";
        if(boosting) sstr_out << "BOOST/";
        else         sstr_out << "NO-BOOST/";
        
        sstr_tr <<  sequence << "_d" << d << "_qp" << QP;
        sstr_out << sequence << "_d" << d << "_qp" << QP << ".out";
        
        trainPath[d] = sstr_tr.str();
        sstr_tr << ".data";
        outPath[d] = sstr_out.str();
        trainFile[d] = fopen(sstr_tr.str().c_str(), "w");
#endif
#if WRITE_TEST
        std::stringstream sstr_te;

        sstr_te << "./TEST/test_" << sequence << "_d" << d << "_qp" << QP << ".data";
        testPath[d] = sstr_te.str();
        testFile[d] = fopen(testPath[d].c_str(), "w");
#endif
    }

}

void TEncDecisionTree::getSADSSE( TComYuv *pcYuvSrc0, TComYuv *pcYuvSrc1,int width){
    const ComponentID compID=ComponentID(COMPONENT_Y);
    UInt uiTrUnitIdx = 0;
    SAD = 0;
    SSE = 0;
    int diff, power2diff;
    const Int uiPartWidth = width >> (pcYuvSrc0->getComponentScaleX(compID));
    const Int uiPartHeight= width >> (pcYuvSrc0->getComponentScaleY(compID));

    Pel* pSrc0 = pcYuvSrc0->getAddr( compID, uiTrUnitIdx, uiPartWidth );
    Pel* pSrc1 = pcYuvSrc1->getAddr( compID, uiTrUnitIdx, uiPartWidth );
          

    Int  iSrc0Stride = pcYuvSrc0->getStride(compID);
    Int  iSrc1Stride = pcYuvSrc1->getStride(compID);

    for (Int y = uiPartHeight-1; y >= 0; y-- )
    {
      for (Int x = uiPartWidth-1; x >= 0; x-- )
      {
        diff = abs(pSrc0[x] - pSrc1[x]);
        power2diff = diff*diff;
        SAD += diff;
        SSE += power2diff;
      }
      pSrc0 += iSrc0Stride;
      pSrc1 += iSrc1Stride;
    }
}

double TEncDecisionTree::getAverageNeighborDepth(TComDataCU *&cu){
    int numValidCtu = 0;
    TComDataCU* upCtu = cu->getCtuAbove();
    TComDataCU* leftCtu = cu->getCtuLeft();
    TComDataCU* upLeftCtu = cu->getCtuAboveLeft();
    TComDataCU* upRightCtu = cu->getCtuAboveRight();
    TComDataCU* colocCtuL0 = cu->getCUColocated(REF_PIC_LIST_0);
    TComDataCU* colocCtuL1 = cu->getCUColocated(REF_PIC_LIST_1);
    if (colocCtuL0 == colocCtuL1)
        colocCtuL1 = NULL;

    double avgCtxDepth = 0;

    if(upCtu){
        numValidCtu++;
        avgCtxDepth += calcAverageDepthIterative(upCtu);
    }
    if(leftCtu){
        numValidCtu++;
        avgCtxDepth += calcAverageDepthIterative(leftCtu);
    }
    if(upLeftCtu){
        numValidCtu++;
        avgCtxDepth += calcAverageDepthIterative(upLeftCtu);
    }
    if(upRightCtu){
        numValidCtu++;
        avgCtxDepth += calcAverageDepthIterative(upRightCtu);
    }
    if(colocCtuL0){
        numValidCtu++;
        avgCtxDepth += calcAverageDepthIterative(colocCtuL0);
    }
    if(colocCtuL1){
        numValidCtu++;
        avgCtxDepth += calcAverageDepthIterative(colocCtuL1);
    }
    return avgCtxDepth/numValidCtu;
}    

double TEncDecisionTree::calcAverageDepthIterative(TComDataCU *&cu){
    int uiDepth,uiAbsPartIdx0,uiAbsPartIdx1,uiAbsPartIdx2;
    UInt uiQNumParts0 ;
    UInt uiQNumParts1 ;
    UInt uiQNumParts2 ;
    UInt uiPartUnitIdx0,uiPartUnitIdx1,uiPartUnitIdx2;
    double avgDepth = 0;

    uiAbsPartIdx0 = 0;
    uiAbsPartIdx1 = 0;
    uiAbsPartIdx2 = 0;
    uiDepth = 0;
    uiQNumParts0 = ( cu->getPic()->getNumPartitionsInCtu() >> (uiDepth<<1) )>>2;
    int totalCus = 0;
    for ( uiPartUnitIdx0 = 0; uiPartUnitIdx0 < 4; uiPartUnitIdx0++, uiAbsPartIdx0+=uiQNumParts0 ){
        if (cu->getDepth(uiAbsPartIdx0) == uiDepth){
            avgDepth += uiDepth;
            totalCus ++;
        }
        else{
            uiDepth = 1;
            uiQNumParts1 = ( cu->getPic()->getNumPartitionsInCtu() >> (uiDepth<<1) )>>2;
            for ( uiAbsPartIdx1 = uiAbsPartIdx0, uiPartUnitIdx1 = 0; uiPartUnitIdx1 < 4; uiPartUnitIdx1++, uiAbsPartIdx1+=uiQNumParts1 ){
                if (cu->getDepth(uiAbsPartIdx1) == uiDepth){
                    avgDepth += uiDepth;
                    totalCus ++;
                }
                else{
                    uiDepth = 2;
                    uiQNumParts2 = ( cu->getPic()->getNumPartitionsInCtu() >> (uiDepth<<1) )>>2;
                    for ( uiAbsPartIdx2 = uiAbsPartIdx1, uiPartUnitIdx2 = 0; uiPartUnitIdx2 < 4; uiPartUnitIdx2++, uiAbsPartIdx2+=uiQNumParts2 ){
                        if (cu->getDepth(uiAbsPartIdx2) == uiDepth){
                            avgDepth += uiDepth;
                            totalCus ++;
                        }
                        else{
                            uiDepth = 3;
                            if (cu->getDepth(uiAbsPartIdx2) == uiDepth){
                                avgDepth += uiDepth;
                                totalCus ++;
                            }
                            else{
                                printf("Error!\n");
                                exit(1);
                            }
                        }
                    }
                }
            }
        }
    }
    
    return avgDepth/totalCus;

}




double TEncDecisionTree::getFeatureValue(TComDataCU *&cu, int feat_idx){
    double feat_val = 0;
    
    int cmode;

    
    if(cu->isSkipped(0)) cmode = 0;
    else if (cu->isInter(0) && !cu->getQtRootCbf( 0 )) cmode = 1;
    else if (cu->isInter(0)) cmode = 2;
    else cmode = 3;

    
    int colocSplit = 0;
    int depth = cu->getDepth(0);
    
    double ratio2Nx2N_MSM = 0,ratioBest_2Nx2N = 0,ratioBest_MSM = 0;
    
    double costBest = cu->getTotalCost();
    if (cost_MSM == 0){
        cost_MSM = 0.1;
    }
   if (cost_2Nx2N == 0)
        cost_2Nx2N = 0.1;
    //if(feat_idx >= 16 && feat_idx <= 20)
    {
        ratio2Nx2N_MSM = cost_2Nx2N / cost_MSM;
        ratioBest_MSM = costBest / cost_MSM;
        ratioBest_2Nx2N = costBest / cost_2Nx2N;
    }
    TComDataCU *colocCuL0 = cu->getCUColocated(REF_PIC_LIST_0);
    TComDataCU *colocCuL1 = cu->getCUColocated(REF_PIC_LIST_1);
    
    if(colocCuL0 == colocCuL1)
        colocCuL1 = NULL;
    
    UInt partIdx = cu->getZorderIdxInCtu();
    if (colocCuL0)
        colocSplit += (int) ((int) (colocCuL0->getDepth(partIdx)) > (int)  (cu->getDepth(0)));
    if (colocCuL1)
        colocSplit += (int) ((int) (colocCuL1->getDepth(partIdx)) > (int)  (cu->getDepth(0)));
    
    int fme = 0;
    int mv_h = 0;
    int mv_v = 0;
    double frac_mv_h = 0;
    double frac_mv_v = 0;
    
    int mvd_h = 0;
    int mvd_v = 0;
    double frac_mvd_h = 0;
    double frac_mvd_v = 0;
    
    int pred_mv_h = 0;
    int pred_mv_v = 0;
    double frac_pred_mv_h = 0;
    double frac_pred_mv_v = 0;
    
    double int_mv_mod = 0;
    double int_pred_mv_mod = 0;
    double int_mvd_mod = 0;
    
    double frac_mv_mod = 0;
    double frac_mvd_mod = 0;
    double frac_pred_mv_mod = 0;
    
    int upSplt =  cu->getAboveSplitFlag( 0, depth );
    int leftSplt =  cu->getLeftSplitFlag( 0, depth ); 
    int upLeftSplt =  cu->getUpLeftSplitFlag( 0, depth ); 
    int upRightSplt =  cu->getUpRightSplitFlag( 0, depth ); 
    int ctxSplit = upSplt + leftSplt + upLeftSplt + upRightSplt + colocSplit;
    
    int uiRefListIdx = 0;
    
    if((feat_idx >= 17 && feat_idx <= 22) || feat_idx == 25){
        for ( uiRefListIdx = 0; uiRefListIdx < 2; uiRefListIdx++ )
        {
            if ( cu->getSlice()->getNumRefIdx( RefPicList( uiRefListIdx ) ) > 0 )
            {
                mv_h = cu->getCUMvField(RefPicList(uiRefListIdx))->getMv( 0 ).getHor();
                mv_v = cu->getCUMvField(RefPicList(uiRefListIdx))->getMv( 0 ).getVer();
                frac_mv_h = (abs(mv_h) % 4)/4.0;
                frac_mv_v = (abs(mv_v) % 4)/4.0;

                mvd_h = cu->getCUMvField(RefPicList(uiRefListIdx))->getMvd( 0 ).getHor();
                mvd_v = cu->getCUMvField(RefPicList(uiRefListIdx))->getMvd( 0 ).getVer();
                frac_mvd_h = (abs(mvd_h) % 4)/4.0;
                frac_mvd_v = (abs(mvd_v) % 4)/4.0;

                pred_mv_h = mvd_h + mv_h;
                pred_mv_v = mvd_v + mv_v;
                frac_pred_mv_h = (abs(pred_mv_h) % 4)/4.0;
                frac_pred_mv_v = (abs(pred_mv_v) % 4)/4.0;

                int_mv_mod = sqrt(((mv_h >> 2) * (mv_h >> 2)) + ((mv_v >> 2) * (mv_v >> 2)));
                int_pred_mv_mod = sqrt(((pred_mv_h >> 2) * (pred_mv_h >> 2)) + ((pred_mv_v >> 2) * (pred_mv_v >> 2)));
                int_mvd_mod = sqrt(((mvd_h >> 2) * (mvd_h >> 2)) + ((mvd_v >> 2) * (mvd_v >> 2)));

                frac_mv_mod = sqrt((frac_mv_h * frac_mv_h) + (frac_mv_v * frac_mv_v));
                frac_mvd_mod = sqrt((frac_mvd_h * frac_mvd_h) + (frac_mvd_v * frac_mvd_v));
                frac_pred_mv_mod = sqrt((frac_pred_mv_h * frac_pred_mv_h) + (frac_pred_mv_v * frac_pred_mv_v));

                if ((frac_mv_v == 0.25) || (frac_mv_h == 0.25)){
                    fme = 2;
                }                
                else if ((frac_mv_v == 0.75) || (frac_mv_h == 0.75)){
                    fme = 2;
                }
                else if ((frac_mv_v == 0.5) || (frac_mv_h == 0.5)){
                    fme = 1;
                }
                else
                    fme = 0;
                break;
            }
        }
    }
    


    
    switch(feat_idx){
        case 1: feat_val =  deltaQP; break;
        case 2: feat_val = cmode; break;
        case 3: feat_val = cu->getPartitionSize(0); break;
        case 4: feat_val = cu->getTotalBits(); break;
        case 5: feat_val = cu->getTotalDistortion(); break;
        case 6: feat_val = cu->getTotalCost(); break;
        case 7: feat_val = SAD; break;
        case 8: feat_val = SSE; break;
        case 9: feat_val =  cost_2Nx2N; break;
        case 10: feat_val =  cost_MSM; break;
        case 11: feat_val =  ratio2Nx2N_MSM; break;
        case 12: feat_val =  ratioBest_2Nx2N; break;
        case 13: feat_val =  ratioBest_MSM; break;
        case 14: feat_val =  cu->getTransformIdx(0); break;
        case 15: feat_val =  nonZeroCoeff; break;
        case 16: feat_val = cu->getCUMvField(RefPicList(uiRefListIdx))->getRefIdx(0) ; break;
        case 17: feat_val = int_mv_mod; break;
        case 18: feat_val = frac_mv_mod; break;
        case 19: feat_val = int_pred_mv_mod; break;
        case 20: feat_val = frac_pred_mv_mod; break;
        case 21: feat_val = int_mvd_mod; break;
        case 22: feat_val = frac_mvd_mod; break;
        case 23: feat_val = cu->getMVPIdx(RefPicList( uiRefListIdx) , 0); break;
        case 24: feat_val = cu->getInterDir(0); break;
        case 25: feat_val = fme; break;
        case 26: feat_val = colocSplit; break;
        case 27: feat_val = ctxSplit; break;
        case 28: feat_val = neighDepth; break;

        default: fprintf(stderr,"Error! Feature IDX %d not supported\n", feat_idx); exit(1) ;
    }
    return feat_val;
}

void TEncDecisionTree::readTree(){
    FILE		*F;
    int			 TotalRules=0;
   /*  Read information on attribute names, values, and classes  */

    FileStem = trainPath[gDepth].c_str();

    if ( ! (F = fopen("hm-feats.names", "r")) ) Error(NOFILE, Fn, "");

    GetNames(F);

    /*  Set up the classification environment  */

    GCEnv[gDepth] = AllocZero(1, CEnvRec);

    GCEnv[gDepth]->ClassWt    = Alloc(MaxClass+1, double);
    GCEnv[gDepth]->Vote       = Alloc(MaxClass+1, float);

    /*  Read the appropriate classifier file.  Call CheckFile() to
	determine the number of trials, then allocate space for
	trees or rulesets  */

    if ( RULES )
    {
	CheckFile(".rules", false);
	RuleSet = AllocZero(TRIALS[gDepth]+1, CRuleSet);

	ForEach(Trial, 0, TRIALS[gDepth]-1)
	{
	    RuleSet[Trial] = GetRules(".rules");
	    TotalRules += RuleSet[Trial]->SNRules;
	}

	if ( RULESUSED )
	{
	    GCEnv[gDepth]->RulesUsed = Alloc(TotalRules + TRIALS[gDepth], RuleNo);
	}

	GCEnv[gDepth]->MostSpec   = Alloc(MaxClass+1, CRule);
    }
    else
    {
	CheckFile(".tree", false);
	Pruned[gDepth] = AllocZero(TRIALS[gDepth]+1, Tree);

	ForEach(Trial, 0, TRIALS[gDepth]-1)
	{
	    Pruned[gDepth][Trial] = GetTree(".tree");
	}
    }

    /*  Close the classifier file and reset the file variable  */
    
    fclose(TRf);
    TRf = 0;

    /*  Set global default class for boosting  */

    Default = ( RULES ? RuleSet[0]->SDefault : Pruned[gDepth][0]->Leaf );

    /*  Now classify the cases in file <filestem>.cases.
	This has the same format as a .data file except that
	the class can be "?" to indicate that it is unknown.  */



}

int TEncDecisionTree::classifyCU(string ddataRec, int depth){
    DataRec		Case;
    ClassNo		Predict;

    void		ShowRules(int);

    //while ( (Case = GetDataRec(dataRec, false)) )
    Case = GetDataRec(ddataRec, false);
    {
	/*  For this case, find the class predicted by See5/C5.0 model  */
	Predict = Classify(Case, GCEnv[depth]);
		
        /*  Free the memory used by this case  */
	FreeLastCase(Case);
    }
    if(GCEnv[depth]->Confidence < 0.9)
        return 1;

    /*  Close the case file and free allocated memory  */
    return atoi(ClassName[Predict]);


}

string TEncDecisionTree::setCUFeatures(TComDataCU *&cu){

    double val;
    int valInt;
   // printf("Mode %d PU %d TrD %d\n", mode, puSize, trDepth);
    int i;
    stringstream sstr;
    
    std::string cu_str = TEncDecisionTree::getMapString(cu, cu->getDepth(0), 0);
    cuOrderMap.push_back(cu_str);

    val = getFeatureValue(cu, 1);
    valInt = (int) val;
    if ( (val - valInt) == 0.0 ) 
        sstr << valInt;
    else
        sstr << val;   
    
    for(i = 2; i <= NB_FEATURES; i++){

        val = getFeatureValue(cu, i); 
        valInt = (int) val;
        if ( (val - valInt) == 0.0 ) 
            sstr << "," << valInt;
        else
            sstr << "," << val;

    }
    if(predictPhase()){
        sstr << ",?\n";
#if WRITE_TEST
        fprintf(testFile[cu->getDepth(0)], "%s\n", sstr.str().c_str() );
#endif
    }
    else if (trainingPhase()){
        statsMap[cu_str] = sstr.str();
    }
    return sstr.str();
}



bool TEncDecisionTree::decideSplit(TComDataCU *&cu, int depth){
    gDepth = depth;
    dataRec = setCUFeatures(cu);
    
    return classifyCU(dataRec, depth);

}



std::string TEncDecisionTree::getMapString(TComDataCU *cu, int depth, int partIdx){
    std::stringstream sstr;
    
    int cu_x = cu->getCUPelX() + g_auiRasterToPelX[ g_auiZscanToRaster[partIdx] ];
    int cu_y = cu->getCUPelY() + g_auiRasterToPelY[ g_auiZscanToRaster[partIdx] ];

    //int cu_x = pcCU->getCUPelX() ;
    //int cu_y = pcCU->getCUPelY() ;
    sstr << cu->getPic()->getPOC() << "_" << cu_x << "x" << cu_y << "_" << depth;
    
    return sstr.str();
}


bool TEncDecisionTree::trainingPhase(){
    if(ONLINE_TRAIN){
        return (encodedFrames >= 4) && (encodedFrames <= (4 + NB_TRAINING_FRAMES));
        
    }
    else{
        return 0;
    }
}

bool TEncDecisionTree::predictPhase(){
    bool enable = false;
    if(ONLINE_TRAIN){
        
        enable = (encodedFrames > (4 + NB_TRAINING_FRAMES));
        if (enable && !trained){
            runC50Train();
            trained = true;
        }
    }
    else{
        enable = encodedFrames >= 4;
    }
    return enable;
}

void TEncDecisionTree::writeTrainFile(){
    //std::map<std::string, std::vector<double> >::iterator it_i;
    //std::vector<double>::iterator it_j;
    
    int i,depth,n_commas;
    for(i = 0; i < cuOrderMap.size(); i++){
        const std::string &s = cuOrderMap[i];
        n_commas = std::count(statsMap[s].begin(), statsMap[s].end(), ',');
        if (n_commas < NB_FEATURES) // skipping CUs below the ones that were not split
            continue;
        depth = s[s.length()-1] - '0';
        
        fprintf(trainFile[depth],"%s\n",statsMap[s].c_str());
        
    }
}

void TEncDecisionTree::runC50Train(){
    int d;

    std::stringstream sstr;
    
    for (d = 0 ; d <= 2; d++){
        sstr.str("");
        fclose(trainFile[d]);
        sstr << "./c5.0 ";
        
        if(boosting) sstr << "-b ";
        
        sstr << "-f " << trainPath[d] <<  " -o " << outPath[d];
        
        printf("%s\n",sstr.str().c_str());
        
        if(system(sstr.str().c_str()) == -1){
            printf("Error training C5.0!\n");
            exit(1);
        }  
        gDepth = d;
        readTree();
    }
}
