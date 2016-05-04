import pandas as pd
from pandas import *
import numpy as np
from numpy import *
import logisticReg
from logisticReg import *

UKCRC = pd.read_stata('MRSA.dta')
MPROS = pd.read_stata('MPROS.dta')

trainData = DataFrame([UKCRC.gen_resist,UKCRC.aad9, UKCRC.aadE, UKCRC.aphA3, UKCRC.ant4_1,
UKCRC.aacA_aphD, UKCRC.aacA, UKCRC.qacA_qacB, UKCRC.smr_qacC,
UKCRC.sdrM, UKCRC.blaZ, UKCRC.blaZ_LGA251, UKCRC.mecA,
UKCRC.fusB, UKCRC.fusC, UKCRC.fusA_any, UKCRC.arsB, UKCRC.arsC,
UKCRC.merABTR, UKCRC.D2JAJ4, UKCRC.cadA, UKCRC.cadC, UKCRC.cadD,
UKCRC.msrA, UKCRC.ermA, UKCRC.ermC, UKCRC.erm_mod_any, UKCRC.ileS_2,
UKCRC.ileS_1_any, UKCRC.grlA_80_4_116_any, UKCRC.grlB_451_any,
UKCRC.gyrA_73_88_any, UKCRC.rpoB_464_484_550_any, UKCRC.tetK,
UKCRC.tetM, UKCRC.dfrA, UKCRC.dfrG, UKCRC.dfrA_B_any])

testData = DataFrame([MPROS.gen_resist, MPROS.aad9, MPROS.aadE, MPROS.aphA3, MPROS.ant4_1,
MPROS.aacA_aphD, MPROS.aacA, MPROS.qacA_qacB, MPROS.smr_qacC,
MPROS.sdrM, MPROS.blaZ, MPROS.blaZ_LGA251, MPROS.mecA,
MPROS.fusB, MPROS.fusC, MPROS.fusA_any, MPROS.arsB, MPROS.arsC,
MPROS.merABTR, MPROS.D2JAJ4, MPROS.cadA, MPROS.cadC, MPROS.cadD,
MPROS.msrA, MPROS.ermA, MPROS.ermC, MPROS.erm_mod_any, MPROS.ileS_2,
MPROS.ileS_1_any, MPROS.grlA_80_4_116_any, MPROS.grlB_451_any,
MPROS.gyrA_73_88_any, MPROS.rpoB_464_484_550_any, MPROS.tetK,
MPROS.tetM, MPROS.dfrA, MPROS.dfrG, MPROS.dfrA_B_any])


trainData = trainData.T.dropna(axis=0)
testData = testData.T.dropna(axis=0)
#print('Dimension of off-NA training dataset is [%d, %d]' % (shape(trainData)[0], shape(trainData)[1]))
#print(trainData.ix[:, 0])

print(testData.isnull().values.any())
trainVals = trainData.ix[:, 1:]
trainLabels = trainData.ix[:, 0]
testVals = testData.ix[:, 1:]
testLabels = testData.ix[:, 0]

GradientIter = 2000

weights = logisticReg.stoGradientAscent(trainVals, trainLabels, GradientIter)




curTestLabel = logisticReg.classifyVector(testVals, weights)

rowTestNum = shape(testVals)[0]
errCnt = 0

for n in range(rowTestNum):
    if int(curTestLabel[n]) is not int(np.array(testLabels)[n]):
        errCnt += 1

accRate = 1 - float(errCnt/rowTestNum)

print(accRate)
