'use strict';

import {Tensor} from '../src/lib/tensor.js';
import * as unaryFunctions from '../src/lib/unary.js';
import * as utils from './utils.js';

describe('test unary', function() {
  function testUnary(op, input, expected, shape) {
    const x = new Tensor(shape, input);
    const y = unaryFunctions[op](x);
    utils.checkShape(y, shape);
    utils.checkValue(y, expected);
  }

  it('abs', function() {
    testUnary('abs', [-1, 0, 1], [1, 0, 1], [3]);
    testUnary(
        'abs',
        [-1.1, 0, 1.1, 2.2, 0, -2.2],
        [1.1, 0, 1.1, 2.2, 0, 2.2],
        [2, 3]);
    testUnary(
        'abs',
        [-1.1, 0, 1.1, 2.2, 0, -2.2],
        [1.1, 0, 1.1, 2.2, 0, 2.2],
        [1, 2, 3]);
    testUnary(
        'abs',
        [-1.1, 0, 1.1, 2.2, 0, -2.2],
        [1.1, 0, 1.1, 2.2, 0, 2.2],
        [1, 2, 3, 1]);
  });

  it('ceil', function() {
    testUnary('ceil', [-1.1, 0, 1.1], [-1, 0, 2], [3]);
    testUnary(
        'ceil',
        [-1.1, 0, 1.1, -2.2, 0, 2.2],
        [-1, 0, 2, -2, 0, 3],
        [2, 3]);
    testUnary(
        'ceil',
        [-1.1, 0, 1.1, -2.2, 0, 2.2],
        [-1, 0, 2, -2, 0, 3],
        [1, 2, 3]);
    testUnary(
        'ceil',
        [-1.1, 0, 1.1, -2.2, 0, 2.2],
        [-1, 0, 2, -2, 0, 3],
        [1, 2, 3, 1]);
  });

  it('cos', function() {
    testUnary(
        'cos',
        [1.4124068, 1.9740626, -0.06506752, 0.73539704],
        [
          0.15772809760857773,
          -0.39242469654349826,
          0.9978838556864368,
          0.7415644450136674,
        ],
        [4]);
    testUnary(
        'cos',
        [
          1.4124068,   1.9740626,  -0.06506752, 0.73539704,
          -0.56439203, 0.89806247, 0.12939146,  -0.34816208,
          -1.0759926,  0.66291636, 0.21504708,  -0.71527237,
        ],
        [
          0.15772809760857773,
          -0.39242469654349826,
          0.9978838556864368,
          0.7415644450136674,
          0.8449139610653698,
          0.6231265199397442,
          0.9916405976730124,
          0.9400013446543031,
          0.4748589278431268,
          0.7882008050685133,
          0.9769663487271947,
          0.7549146426895217,
        ],
        [3, 4]);
    testUnary(
        'cos',
        [
          1.4124068,   1.9740626,
          -0.06506752, 0.73539704,
          -0.56439203, 0.89806247,
          0.12939146,  -0.34816208,
          -1.0759926,  0.66291636,
          0.21504708,  -0.71527237,
        ],
        [
          0.15772809760857773,
          -0.39242469654349826,
          0.9978838556864368,
          0.7415644450136674,
          0.8449139610653698,
          0.6231265199397442,
          0.9916405976730124,
          0.9400013446543031,
          0.4748589278431268,
          0.7882008050685133,
          0.9769663487271947,
          0.7549146426895217,
        ],
        [3, 2, 2]);
    testUnary(
        'cos',
        [
          1.4124068,
          1.9740626,
          -0.06506752,
          0.73539704,
          -0.56439203,
          0.89806247,
          0.12939146,
          -0.34816208,
          -1.0759926,
          0.66291636,
          0.21504708,
          -0.71527237,
        ],
        [
          0.15772809760857773,
          -0.39242469654349826,
          0.9978838556864368,
          0.7415644450136674,
          0.8449139610653698,
          0.6231265199397442,
          0.9916405976730124,
          0.9400013446543031,
          0.4748589278431268,
          0.7882008050685133,
          0.9769663487271947,
          0.7549146426895217,
        ],
        [3, 2, 2, 1]);
  });

  it('exp', function() {
    testUnary('exp', [-1, 0, 1], [0.36787944117144233, 1, 2.718281828459045], [3]);
    testUnary(
        'exp',
        [
          0.3143407, 0.03632548, 0.5354084,  -0.5000897,
          1.2028517, -1.2581364, -1.5108215, -1.2340564,
          1.3860914, -0.2944251, -1.5065757, -0.4673513,
        ],
        [
          1.3693561967375985,
          1.036993312152022,
          1.708145706528859,
          0.6064762563524844,
          3.3295984129224556,
          0.2841831370156004,
          0.22072857493397907,
          0.29110932356722374,
          3.999188237901296,
          0.7449597417366962,
          0.2216677366530061,
          0.6266599061142515,
        ],
        [3, 4]);
    testUnary(
        'exp',
        [
          0.3143407,   0.03632548,  0.5354084,   -0.5000897,  1.2028517,
          -1.2581364,  -1.5108215,  -1.2340564,  1.3860914,   -0.2944251,
          -1.5065757,  -0.4673513,  0.56616277,  0.77866685,  -0.01097398,
          1.0758846,   0.6035437,   0.36806744,  0.03906458,  -0.54385495,
          0.10609569,  -0.40644982, -1.2890846,  1.3825086,   0.51489764,
          1.6407244,   -0.67886734, -0.6556329,  1.0399923,   0.1484657,
          1.011217,    0.8451463,   0.75473833,  -2.0161264,  1.6406634,
          -0.01692923, -0.7986609,  0.97758174,  0.893054,    -0.01632686,
          -1.9721986,  -0.75843745, 0.42327842,  -0.08648382, -1.3960054,
          0.7547995,   -0.42002508, -1.784105,   1.0171342,   0.3634587,
          0.4158588,   -1.0103701,  -0.23202766, 0.6390487,   -0.22796124,
          0.11259284,  0.3690759,   -0.18703128, 0.07711394,  2.9116163,
        ],
        [
          1.3693561967375985,
          1.036993312152022,
          1.708145706528859,
          0.6064762563524844,
          3.3295984129224556,
          0.2841831370156004,
          0.22072857493397907,
          0.29110932356722374,
          3.999188237901296,
          0.7449597417366962,
          0.2216677366530061,
          0.6266599061142515,
          1.7614948056979465,
          2.1785659734395244,
          0.9890860144586422,
          2.9325859189764807,
          1.828587297220383,
          1.4449394824067432,
          1.0398376341962599,
          0.580506111445723,
          1.111928271832297,
          0.666010515185213,
          0.27552288133235436,
          3.984885583357506,
          1.6734671969280184,
          5.158905269957428,
          0.5071911422650927,
          0.5191134117181747,
          2.829195229464421,
          1.1600530072738744,
          2.7489444455169916,
          2.328318422639379,
          2.127054864081674,
          0.13317031580877914,
          5.158590586333908,
          0.9832132641755142,
          0.4499310635787923,
          2.6580206785468157,
          2.4425779049391343,
          0.983805700764556,
          0.13915058318029658,
          0.4683977504013186,
          1.526959372817342,
          0.9171503881579351,
          0.2475839902489274,
          2.127184980007265,
          0.6570303412874574,
          0.16794730659465534,
          2.765258719399421,
          1.4382954540763537,
          1.5156718408966148,
          0.36408420706837397,
          0.7929241907202421,
          1.8946776149031732,
          0.7961551182097963,
          1.119176156404052,
          1.446397381069858,
          0.8294177917911054,
          1.0801651433977786,
          18.38649265135158,
        ],
        [3, 4, 5]);
    testUnary(
        'exp',
        [
          0.3143407,   0.03632548,  0.5354084,   -0.5000897,  1.2028517,
          -1.2581364,  -1.5108215,  -1.2340564,  1.3860914,   -0.2944251,
          -1.5065757,  -0.4673513,  0.56616277,  0.77866685,  -0.01097398,
          1.0758846,   0.6035437,   0.36806744,  0.03906458,  -0.54385495,
          0.10609569,  -0.40644982, -1.2890846,  1.3825086,   0.51489764,
          1.6407244,   -0.67886734, -0.6556329,  1.0399923,   0.1484657,
          1.011217,    0.8451463,   0.75473833,  -2.0161264,  1.6406634,
          -0.01692923, -0.7986609,  0.97758174,  0.893054,    -0.01632686,
          -1.9721986,  -0.75843745, 0.42327842,  -0.08648382, -1.3960054,
          0.7547995,   -0.42002508, -1.784105,   1.0171342,   0.3634587,
          0.4158588,   -1.0103701,  -0.23202766, 0.6390487,   -0.22796124,
          0.11259284,  0.3690759,   -0.18703128, 0.07711394,  2.9116163,
        ],
        [
          1.3693561967375985,
          1.036993312152022,
          1.708145706528859,
          0.6064762563524844,
          3.3295984129224556,
          0.2841831370156004,
          0.22072857493397907,
          0.29110932356722374,
          3.999188237901296,
          0.7449597417366962,
          0.2216677366530061,
          0.6266599061142515,
          1.7614948056979465,
          2.1785659734395244,
          0.9890860144586422,
          2.9325859189764807,
          1.828587297220383,
          1.4449394824067432,
          1.0398376341962599,
          0.580506111445723,
          1.111928271832297,
          0.666010515185213,
          0.27552288133235436,
          3.984885583357506,
          1.6734671969280184,
          5.158905269957428,
          0.5071911422650927,
          0.5191134117181747,
          2.829195229464421,
          1.1600530072738744,
          2.7489444455169916,
          2.328318422639379,
          2.127054864081674,
          0.13317031580877914,
          5.158590586333908,
          0.9832132641755142,
          0.4499310635787923,
          2.6580206785468157,
          2.4425779049391343,
          0.983805700764556,
          0.13915058318029658,
          0.4683977504013186,
          1.526959372817342,
          0.9171503881579351,
          0.2475839902489274,
          2.127184980007265,
          0.6570303412874574,
          0.16794730659465534,
          2.765258719399421,
          1.4382954540763537,
          1.5156718408966148,
          0.36408420706837397,
          0.7929241907202421,
          1.8946776149031732,
          0.7961551182097963,
          1.119176156404052,
          1.446397381069858,
          0.8294177917911054,
          1.0801651433977786,
          18.38649265135158,
        ],
        [3, 2, 2, 5]);
  });

  it('floor', function() {
    testUnary('floor', [-1.1, 0, 1.1], [-2, 0, 1], [3]);
    testUnary(
        'floor',
        [-1.1, 0, 1.1, -2.2, 0, 2.2],
        [-2, 0, 1, -3, 0, 2],
        [2, 3]);
    testUnary(
        'floor',
        [-1.1, 0, 1.1, -2.2, 0, 2.2],
        [-2, 0, 1, -3, 0, 2],
        [1, 2, 3]);
    testUnary(
        'floor',
        [-1.1, 0, 1.1, -2.2, 0, 2.2],
        [-2, 0, 1, -3, 0, 2],
        [1, 2, 3, 1]);
  });

  it('log', function() {
    testUnary(
        'log',
        [1.4599811, 0.34325936, 1.0420732],
        [
          0.37842349043097573,
          -1.0692689659512902,
          0.04121219038394666,
        ],
        [3]);
    testUnary(
        'log',
        [
          1.4599811,  0.34325936, 1.0420732, 0.10867598,
          0.39999306, 0.03704359, 1.5873954, 0.44784936,
          0.69070333, 1.8561625,  1.4088289, 0.06367786,
        ],
        [
          0.37842349043097573,
          -1.0692689659512902,
          0.04121219038394666,
          -2.219384484434575,
          -0.9163080820246681,
          -3.2956599516545957,
          0.4620945598501042,
          -0.8032983531118589,
          -0.3700448817029436,
          0.618511184410702,
          0.34275879190200165,
          -2.753918343538326,
        ],
        [3, 4]);
    testUnary(
        'log',
        [
          1.4599811,  0.34325936, 1.0420732,  0.10867598, 0.39999306,
          0.03704359, 1.5873954,  0.44784936, 0.69070333, 1.8561625,
          1.4088289,  0.06367786, 0.32938832, 1.2429568,  1.1544572,
          0.47578564, 1.868428,   1.2279319,  1.0712656,  1.17982,
          1.460244,   0.62389,    0.79644215, 0.4196875,  0.372386,
          1.8887448,  1.4791015,  0.98091763, 0.45482925, 0.50871295,
          0.11605832, 0.86883324, 0.6235918,  1.392687,   0.75550365,
          0.35920736, 0.04935746, 0.13449927, 1.3587855,  0.9073937,
          1.0731584,  1.7933426,  1.9806778,  0.43379396, 1.3261564,
          0.52664477, 0.041302,   1.5167572,  0.6400343,  0.7669278,
          1.1766342,  1.6620969,  1.2579637,  1.7453014,  0.5470841,
          1.5960937,  0.37127188, 1.9055833,  1.3749765,  0.43101534,
        ],
        [
          0.37842349043097573,
          -1.0692689659512902,
          0.04121219038394666,
          -2.219384484434575,
          -0.9163080820246681,
          -3.2956599516545957,
          0.4620945598501042,
          -0.8032983531118589,
          -0.3700448817029436,
          0.618511184410702,
          0.34275879190200165,
          -2.753918343538326,
          -1.1105179202764903,
          0.21749305729871296,
          0.1436302767995352,
          -0.7427878623169412,
          0.6250974356178759,
          0.2053313721611498,
          0.0688407532508926,
          0.16536188446892103,
          0.3786035450443755,
          -0.47178120820349856,
          -0.22760078252711477,
          -0.8682448922645803,
          -0.987824328270859,
          0.6359124814574089,
          0.3914348088248874,
          -0.019266788283548882,
          -0.7878332051896428,
          -0.6758713704300671,
          -2.1536624555958537,
          -0.14060407086584029,
          -0.4722592913397491,
          0.3312349746469,
          -0.2803706660434222,
          -1.023855452786281,
          -3.0086663593800362,
          -2.006196507464266,
          0.306591286066901,
          -0.09717885469019541,
          0.0706060762388415,
          0.5840812527784782,
          0.6834391093595378,
          -0.835185604153331,
          0.2822848335262163,
          -0.6412290184429051,
          -3.1868443540375373,
          0.41657463482092677,
          -0.4462335103145133,
          -0.26536261503132674,
          0.162657989828439,
          0.5080799979827856,
          0.22949430253601164,
          0.5569272628026896,
          -0.6031527406633165,
          0.467559206577478,
          -0.990820654574951,
          0.644788155936501,
          0.31843664006339245,
          -0.8416115978644251,
        ],
        [3, 4, 5]);
    testUnary(
        'log',
        [
          1.4599811,  0.34325936, 1.0420732,  0.10867598, 0.39999306,
          0.03704359, 1.5873954,  0.44784936, 0.69070333, 1.8561625,
          1.4088289,  0.06367786, 0.32938832, 1.2429568,  1.1544572,
          0.47578564, 1.868428,   1.2279319,  1.0712656,  1.17982,
          1.460244,   0.62389,    0.79644215, 0.4196875,  0.372386,
          1.8887448,  1.4791015,  0.98091763, 0.45482925, 0.50871295,
          0.11605832, 0.86883324, 0.6235918,  1.392687,   0.75550365,
          0.35920736, 0.04935746, 0.13449927, 1.3587855,  0.9073937,
          1.0731584,  1.7933426,  1.9806778,  0.43379396, 1.3261564,
          0.52664477, 0.041302,   1.5167572,  0.6400343,  0.7669278,
          1.1766342,  1.6620969,  1.2579637,  1.7453014,  0.5470841,
          1.5960937,  0.37127188, 1.9055833,  1.3749765,  0.43101534,
        ],
        [
          0.37842349043097573,
          -1.0692689659512902,
          0.04121219038394666,
          -2.219384484434575,
          -0.9163080820246681,
          -3.2956599516545957,
          0.4620945598501042,
          -0.8032983531118589,
          -0.3700448817029436,
          0.618511184410702,
          0.34275879190200165,
          -2.753918343538326,
          -1.1105179202764903,
          0.21749305729871296,
          0.1436302767995352,
          -0.7427878623169412,
          0.6250974356178759,
          0.2053313721611498,
          0.0688407532508926,
          0.16536188446892103,
          0.3786035450443755,
          -0.47178120820349856,
          -0.22760078252711477,
          -0.8682448922645803,
          -0.987824328270859,
          0.6359124814574089,
          0.3914348088248874,
          -0.019266788283548882,
          -0.7878332051896428,
          -0.6758713704300671,
          -2.1536624555958537,
          -0.14060407086584029,
          -0.4722592913397491,
          0.3312349746469,
          -0.2803706660434222,
          -1.023855452786281,
          -3.0086663593800362,
          -2.006196507464266,
          0.306591286066901,
          -0.09717885469019541,
          0.0706060762388415,
          0.5840812527784782,
          0.6834391093595378,
          -0.835185604153331,
          0.2822848335262163,
          -0.6412290184429051,
          -3.1868443540375373,
          0.41657463482092677,
          -0.4462335103145133,
          -0.26536261503132674,
          0.162657989828439,
          0.5080799979827856,
          0.22949430253601164,
          0.5569272628026896,
          -0.6031527406633165,
          0.467559206577478,
          -0.990820654574951,
          0.644788155936501,
          0.31843664006339245,
          -0.8416115978644251,
        ],
        [3, 2, 2, 5]);
  });

  it('neg', function() {
    testUnary('neg', [-1.1, 0, 1.1], [1.1, -0, -1.1], [3]);
    testUnary(
        'neg',
        [-1, 0, 1.1, -2.2, 0, 2],
        [1, -0, -1.1, 2.2, -0, -2],
        [2, 3]);
    testUnary(
        'neg',
        [-1, 0, 1.1, -2.2, 0, 2],
        [1, -0, -1.1, 2.2, -0, -2],
        [1, 2, 3]);
    testUnary(
        'neg',
        [-1, 0, 1.1, -2.2, 0, 2],
        [1, -0, -1.1, 2.2, -0, -2],
        [1, 2, 3, 1]);
  });

  it('sin', function() {
    testUnary(
        'sin',
        [1.4124068, 1.9740626, -0.06506752, 0.73539704],
        [
          0.9874825807196697,
          0.9197841363835013,
          -0.06502161610088251,
          0.6708816392565617,
        ],
        [4]);
    testUnary(
        'sin',
        [
          1.4124068,   1.9740626,  -0.06506752, 0.73539704,
          -0.56439203, 0.89806247, 0.12939146,  -0.34816208,
          -1.0759926,  0.66291636, 0.21504708,  -0.71527237,
        ],
        [
          0.9874825807196697,
          0.9197841363835013,
          -0.06502161610088251,
          0.6708816392565617,
          -0.5349022325592097,
          0.7821210521062475,
          0.12903071357901844,
          -0.34117073738540654,
          -0.8800619288707335,
          0.6154181431265636,
          0.21339342411295945,
          -0.6558230571220807,
        ],
        [3, 4]);
    testUnary(
        'sin',
        [
          1.4124068,   1.9740626,
          -0.06506752, 0.73539704,
          -0.56439203, 0.89806247,
          0.12939146,  -0.34816208,
          -1.0759926,  0.66291636,
          0.21504708,  -0.71527237,
        ],
        [
          0.9874825807196697,
          0.9197841363835013,
          -0.06502161610088251,
          0.6708816392565617,
          -0.5349022325592097,
          0.7821210521062475,
          0.12903071357901844,
          -0.34117073738540654,
          -0.8800619288707335,
          0.6154181431265636,
          0.21339342411295945,
          -0.6558230571220807,
        ],
        [3, 2, 2]);
    testUnary(
        'sin',
        [
          1.4124068,
          1.9740626,
          -0.06506752,
          0.73539704,
          -0.56439203,
          0.89806247,
          0.12939146,
          -0.34816208,
          -1.0759926,
          0.66291636,
          0.21504708,
          -0.71527237,
        ],
        [
          0.9874825807196697,
          0.9197841363835013,
          -0.06502161610088251,
          0.6708816392565617,
          -0.5349022325592097,
          0.7821210521062475,
          0.12903071357901844,
          -0.34117073738540654,
          -0.8800619288707335,
          0.6154181431265636,
          0.21339342411295945,
          -0.6558230571220807,
        ],
        [3, 2, 2, 1]);
  });

  it('tan', function() {
    testUnary(
        'tan',
        [1.4124068, 1.9740626, -0.06506752, 0.73539704],
        [
          6.260663735197218,
          -2.3438487548949354,
          -0.06515950301265735,
          0.9046842034668978,
        ],
        [4]);
    testUnary(
        'tan',
        [
          1.4124068,   1.9740626,  -0.06506752, 0.73539704,
          -0.56439203, 0.89806247, 0.12939146,  -0.34816208,
          -1.0759926,  0.66291636, 0.21504708,  -0.71527237,
        ],
        [
          6.260663735197218,
          -2.3438487548949354,
          -0.06515950301265735,
          0.9046842034668978,
          -0.633084855036293,
          1.2551560992491184,
          0.1301184258508601,
          -0.3629470737734702,
          -1.8533123782006025,
          0.7807885239004154,
          0.2184245387683735,
          -0.8687380268391548,
        ],
        [3, 4]);
    testUnary(
        'tan',
        [
          1.4124068,   1.9740626,
          -0.06506752, 0.73539704,
          -0.56439203, 0.89806247,
          0.12939146,  -0.34816208,
          -1.0759926,  0.66291636,
          0.21504708,  -0.71527237,
        ],
        [
          6.260663735197218,
          -2.3438487548949354,
          -0.06515950301265735,
          0.9046842034668978,
          -0.633084855036293,
          1.2551560992491184,
          0.1301184258508601,
          -0.3629470737734702,
          -1.8533123782006025,
          0.7807885239004154,
          0.2184245387683735,
          -0.8687380268391548,
        ],
        [3, 2, 2]);
    testUnary(
        'tan',
        [
          1.4124068,
          1.9740626,
          -0.06506752,
          0.73539704,
          -0.56439203,
          0.89806247,
          0.12939146,
          -0.34816208,
          -1.0759926,
          0.66291636,
          0.21504708,
          -0.71527237,
        ],
        [
          6.260663735197218,
          -2.3438487548949354,
          -0.06515950301265735,
          0.9046842034668978,
          -0.633084855036293,
          1.2551560992491184,
          0.1301184258508601,
          -0.3629470737734702,
          -1.8533123782006025,
          0.7807885239004154,
          0.2184245387683735,
          -0.8687380268391548,
        ],
        [3, 2, 2, 1]);
  });
});
