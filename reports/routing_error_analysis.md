# Routing Error Analysis

Development analysis at a fixed 30% budget (97/324 queries). Logistic scores estimate positive-gain probability; tree scores estimate continuous gain. They are not interchangeable scales. Candidate coverage and actual gains below are post-hoc diagnostics, never model inputs.

## Selection outcomes

| Policy | Positive | Zero | Negative | No judged positive candidate |
|---|---:|---:|---:|---:|
| disagreement | 35 | 51 | 11 | 40 |
| logistic | 39 | 24 | 34 | 9 |
| regression_tree | 33 | 48 | 16 | 39 |

Logistic regression selects more positive-gain queries than disagreement, but also many more negative-gain queries. Positive-query count alone does not measure the quality of budget allocation. The tree selects a similar number of candidate-empty queries to disagreement; that count alone does not explain its deficit. These observations describe outcomes, not established causes of model failure.

The fixed regression tree gives zero split importance to retrieval Jaccard overlap in its full-training fit, while the disagreement baseline ranks directly by it. This is a limitation of this fitted shallow tree and objective; it is not evidence that overlap lacks useful information. The available experiment does not distinguish insufficient features, limited sample size, or unsuitable inductive bias.

## logistic

Selected outcomes: 39 positive, 24 zero, 34 negative.

Unique predicted scores: 324. Budget-boundary score has 1 tied queries, of which 1 are selected by query-ID order. Learned-only selections: 93, total realized gain +0.931412; disagreement-only selections: 93, total realized gain +3.757654. Divide the gain-sum difference by 324 for the policy NDCG difference.

### Highest predicted scores (20 queries)

| Query ID | Query | Predicted score | Actual gain | Base NDCG | Judged positives in pool |
|---|---|---:|---:|---:|---:|
| PLAIN-174 | Cinnamon for Diabetes | 0.531913 | -0.006566 | 0.866948 | 11 |
| PLAIN-3160 | Egg Cholesterol in the Diet | 0.494726 | +0.150661 | 0.164220 | 3 |
| PLAIN-2889 | Extra Virgin Olive Oil vs. Nuts | 0.483378 | +0.500000 | 0.500000 | 1 |
| PLAIN-1261 | glaucoma | 0.469323 | -0.016390 | 0.648932 | 5 |
| PLAIN-2599 | Alpha Gal and the Lone Star Tick | 0.467618 | +0.063621 | 0.936379 | 13 |
| PLAIN-417 | Does wild rice contain the same levels of arsenic as brown rice? | 0.467406 | +0.169580 | 0.529635 | 5 |
| PLAIN-3411 | Dairy & Acne | 0.462372 | -0.216159 | 0.926636 | 12 |
| PLAIN-2649 | Boosting the Bioavailability of Curcumin | 0.459388 | -0.094788 | 0.498290 | 4 |
| PLAIN-1907 | porcine endogenous retroviruses | 0.458550 | +0.069130 | 0.681206 | 7 |
| PLAIN-3461 | Coffee and Mortality | 0.457623 | +0.000000 | 0.760079 | 11 |
| PLAIN-1287 | grapefruit juice | 0.455645 | -0.135685 | 1.000000 | 13 |
| PLAIN-1428 | iodine | 0.449709 | +0.000000 | 1.000000 | 14 |
| PLAIN-43 | Organic Milk and Prostate Cancer | 0.448798 | -0.016390 | 0.563788 | 5 |
| PLAIN-2239 | tick bites | 0.442425 | -0.046143 | 0.384311 | 5 |
| PLAIN-1918 | poultry viruses | 0.440226 | -0.188444 | 0.800694 | 7 |
| PLAIN-1129 | erectile dysfunction | 0.435679 | -0.021522 | 0.936379 | 19 |
| PLAIN-2133 | spirulina | 0.431934 | +0.000000 | 0.693022 | 5 |
| PLAIN-3371 | A Harmless Artificial Sweetener | 0.430249 | +0.145626 | 0.217261 | 3 |
| PLAIN-612 | asbestos | 0.428765 | +0.000000 | 0.000000 | 0 |
| PLAIN-2679 | Phosphate Additives in Meat Purge and Cola | 0.427922 | +0.066254 | 0.864315 | 15 |

### Most harmful selected queries

| Query ID | Query | Predicted score | Actual gain | Base NDCG | Judged positives in pool |
|---|---|---:|---:|---:|---:|
| PLAIN-3411 | Dairy & Acne | 0.462372 | -0.216159 | 0.926636 | 12 |
| PLAIN-2290 | ulcerative colitis | 0.418232 | -0.215242 | 0.616367 | 8 |
| PLAIN-2789 | Protein, Puberty, and Pollutants | 0.366580 | -0.212502 | 0.708441 | 6 |
| PLAIN-1761 | oral health | 0.357675 | -0.198187 | 0.936379 | 16 |
| PLAIN-1918 | poultry viruses | 0.440226 | -0.188444 | 0.800694 | 7 |
| PLAIN-2342 | vitamin E | 0.367250 | -0.176018 | 0.494357 | 6 |
| PLAIN-1192 | flavonols | 0.415643 | -0.173187 | 0.173187 | 2 |
| PLAIN-2270 | tryptophan | 0.374580 | -0.150431 | 0.654809 | 3 |
| PLAIN-3200 | Anabolic Steroids in Meat | 0.378892 | -0.137083 | 0.586116 | 6 |
| PLAIN-1287 | grapefruit juice | 0.455645 | -0.135685 | 1.000000 | 13 |
| PLAIN-2759 | Breast Cancer and Alcohol: How Much is Safe? | 0.408179 | -0.131753 | 0.478704 | 5 |
| PLAIN-3240 | Is Homeopathy Just Placebo? | 0.383006 | -0.129484 | 0.848603 | 11 |
| PLAIN-1896 | polyphenols | 0.394387 | -0.112447 | 0.790951 | 11 |
| PLAIN-1017 | detoxification | 0.377570 | -0.110046 | 0.220092 | 3 |
| PLAIN-2649 | Boosting the Bioavailability of Curcumin | 0.459388 | -0.094788 | 0.498290 | 4 |
| PLAIN-1160 | fatty liver disease | 0.403707 | -0.094788 | 0.094788 | 3 |
| PLAIN-3421 | Broccoli Sprouts | 0.401171 | -0.090119 | 0.711618 | 8 |
| PLAIN-2385 | World Health Organization | 0.376044 | -0.081210 | 0.154574 | 3 |
| PLAIN-2659 | Greens vs. Glaucoma | 0.381289 | -0.073117 | 0.721056 | 6 |
| PLAIN-2609 | Treating Asthma and Eczema With Plant-Based Diets | 0.398854 | -0.066254 | 0.811556 | 13 |

### Largest missed gains

| Query ID | Query | Predicted score | Actual gain | Base NDCG | Judged positives in pool |
|---|---|---:|---:|---:|---:|
| PLAIN-3084 | How the Institute of Medicine Arrived at Their Vitamin D Recommendation | 0.214339 | +0.500000 | 0.000000 | 1 |
| PLAIN-3341 | Are Avocados Good for You? | 0.332356 | +0.411371 | 0.174371 | 3 |
| PLAIN-2091 | sinusitis | 0.266628 | +0.403836 | 0.543067 | 3 |
| PLAIN-2229 | thermogenics | 0.262685 | +0.312852 | 0.156426 | 1 |
| PLAIN-2208 | tapeworms | 0.265397 | +0.310493 | 0.224761 | 4 |
| PLAIN-1578 | mastectomy | 0.309888 | +0.289043 | 0.422575 | 6 |
| PLAIN-1108 | endocannabinoids | 0.302523 | +0.287737 | 0.416181 | 2 |
| PLAIN-1971 | quercitin | 0.240372 | +0.273811 | 0.085143 | 2 |
| PLAIN-3270 | Flaxseeds vs. Chia Seeds | 0.293729 | +0.273587 | 0.000000 | 4 |
| PLAIN-3130 | Are Avocados Bad for You? | 0.338170 | +0.235349 | 0.094788 | 2 |
| PLAIN-152 | How Probiotics Affect Mental Health | 0.316254 | +0.230474 | 0.769526 | 13 |
| PLAIN-2979 | Cow’s Milk Casomorphin and Crib Death | 0.329826 | +0.226294 | 0.386853 | 1 |
| PLAIN-1620 | methylmalonic acid | 0.323025 | +0.221282 | 0.383649 | 2 |
| PLAIN-3210 | Caffeine During Pregnancy | 0.312923 | +0.201657 | 0.438287 | 9 |
| PLAIN-690 | bio identical hormones | 0.281824 | +0.194813 | 0.107789 | 1 |
| PLAIN-1740 | nutritional yeast | 0.337996 | +0.182654 | 0.371489 | 4 |
| PLAIN-2990 | Watermelon as Treatment for Erectile Dysfunction | 0.351446 | +0.180091 | 0.139056 | 2 |
| PLAIN-2569 | Prevent Cancer From Going on TOR | 0.259408 | +0.179957 | 0.312049 | 4 |
| PLAIN-1515 | limonin | 0.206284 | +0.172279 | 0.130324 | 1 |
| PLAIN-2809 | Apple Skin: Peeling Back Cancer | 0.316914 | +0.166596 | 0.397192 | 4 |

### Learned-only selections

| Query ID | Query | Predicted score | Actual gain | Base NDCG | Judged positives in pool |
|---|---|---:|---:|---:|---:|
| PLAIN-3411 | Dairy & Acne | 0.462372 | -0.216159 | 0.926636 | 12 |
| PLAIN-2290 | ulcerative colitis | 0.418232 | -0.215242 | 0.616367 | 8 |
| PLAIN-2789 | Protein, Puberty, and Pollutants | 0.366580 | -0.212502 | 0.708441 | 6 |
| PLAIN-1761 | oral health | 0.357675 | -0.198187 | 0.936379 | 16 |
| PLAIN-1918 | poultry viruses | 0.440226 | -0.188444 | 0.800694 | 7 |
| PLAIN-2342 | vitamin E | 0.367250 | -0.176018 | 0.494357 | 6 |
| PLAIN-1192 | flavonols | 0.415643 | -0.173187 | 0.173187 | 2 |
| PLAIN-2270 | tryptophan | 0.374580 | -0.150431 | 0.654809 | 3 |
| PLAIN-3200 | Anabolic Steroids in Meat | 0.378892 | -0.137083 | 0.586116 | 6 |
| PLAIN-1287 | grapefruit juice | 0.455645 | -0.135685 | 1.000000 | 13 |
| PLAIN-2759 | Breast Cancer and Alcohol: How Much is Safe? | 0.408179 | -0.131753 | 0.478704 | 5 |
| PLAIN-3240 | Is Homeopathy Just Placebo? | 0.383006 | -0.129484 | 0.848603 | 11 |
| PLAIN-1896 | polyphenols | 0.394387 | -0.112447 | 0.790951 | 11 |
| PLAIN-2649 | Boosting the Bioavailability of Curcumin | 0.459388 | -0.094788 | 0.498290 | 4 |
| PLAIN-1160 | fatty liver disease | 0.403707 | -0.094788 | 0.094788 | 3 |
| PLAIN-3421 | Broccoli Sprouts | 0.401171 | -0.090119 | 0.711618 | 8 |
| PLAIN-2659 | Greens vs. Glaucoma | 0.381289 | -0.073117 | 0.721056 | 6 |
| PLAIN-2609 | Treating Asthma and Eczema With Plant-Based Diets | 0.398854 | -0.066254 | 0.811556 | 13 |
| PLAIN-2849 | Protein and Heart Disease | 0.386919 | -0.048029 | 0.146068 | 1 |
| PLAIN-2239 | tick bites | 0.442425 | -0.046143 | 0.384311 | 5 |
| PLAIN-2509 | Chicken Big: Poultry and Obesity | 0.396529 | -0.035279 | 0.608317 | 7 |
| PLAIN-1129 | erectile dysfunction | 0.435679 | -0.021522 | 0.936379 | 19 |
| PLAIN-1352 | herbs | 0.388920 | -0.019868 | 0.357076 | 6 |
| PLAIN-1261 | glaucoma | 0.469323 | -0.016390 | 0.648932 | 5 |
| PLAIN-43 | Organic Milk and Prostate Cancer | 0.448798 | -0.016390 | 0.563788 | 5 |
| PLAIN-1589 | medical education | 0.379353 | -0.012535 | 0.529898 | 7 |
| PLAIN-3331 | Salt OK if Blood Pressure is OK? | 0.392584 | -0.012144 | 0.078398 | 1 |
| PLAIN-1038 | dogs | 0.369112 | -0.009447 | 0.769526 | 8 |
| PLAIN-1245 | gastric emptying | 0.379616 | -0.007110 | 0.293456 | 2 |
| PLAIN-174 | Cinnamon for Diabetes | 0.531913 | -0.006566 | 0.866948 | 11 |
| PLAIN-90 | How to Boost the Benefits of Exercise | 0.372368 | -0.000934 | 0.215018 | 7 |
| PLAIN-1027 | dietary guidelines | 0.394776 | -0.000678 | 0.586795 | 8 |
| PLAIN-101 | How to Treat Multiple Sclerosis With Diet | 0.393279 | +0.000000 | 0.563788 | 5 |
| PLAIN-1140 | European Food Safety Authority | 0.378764 | +0.000000 | 0.000000 | 0 |
| PLAIN-1341 | HeLa cells | 0.400104 | +0.000000 | 0.000000 | 0 |
| PLAIN-1418 | insecticides | 0.423383 | +0.000000 | 0.000000 | 0 |
| PLAIN-1428 | iodine | 0.449709 | +0.000000 | 1.000000 | 14 |
| PLAIN-1939 | processed meats | 0.385290 | +0.000000 | 0.000000 | 0 |
| PLAIN-2070 | sense of smell | 0.369687 | +0.000000 | 0.000000 | 0 |
| PLAIN-2133 | spirulina | 0.431934 | +0.000000 | 0.693022 | 5 |
| PLAIN-216 | Plant-Based Diets for Metabolic Syndrome | 0.387797 | +0.000000 | 0.000000 | 0 |
| PLAIN-2219 | telomeres | 0.418701 | +0.000000 | 1.000000 | 16 |
| PLAIN-2280 | turmeric | 0.387389 | +0.000000 | 1.000000 | 20 |
| PLAIN-2353 | walnut | 0.410670 | +0.000000 | 0.000000 | 1 |
| PLAIN-2699 | Treating Multiple Sclerosis With the Swank MS Diet | 0.374598 | +0.000000 | 0.563788 | 4 |
| PLAIN-3220 | Reversing Cognitive Decline | 0.379691 | +0.000000 | 0.000000 | 1 |
| PLAIN-3351 | Update on Spirulina | 0.366207 | +0.000000 | 0.430677 | 1 |
| PLAIN-3381 | Are Artificial Colors Harmful? | 0.359204 | +0.000000 | 0.000000 | 0 |
| PLAIN-3461 | Coffee and Mortality | 0.457623 | +0.000000 | 0.760079 | 11 |
| PLAIN-440 | What's the right dose of saffron? | 0.381254 | +0.000000 | 0.933746 | 11 |
| PLAIN-519 | allergies | 0.382990 | +0.000000 | 1.000000 | 18 |
| PLAIN-530 | ALS | 0.414434 | +0.000000 | 1.000000 | 17 |
| PLAIN-612 | asbestos | 0.428765 | +0.000000 | 0.000000 | 0 |
| PLAIN-656 | Barrett's esophagus | 0.383047 | +0.000000 | 0.000000 | 0 |
| PLAIN-740 | BPA | 0.392603 | +0.000000 | 0.804810 | 3 |
| PLAIN-859 | cellulite | 0.367704 | +0.000000 | 0.539003 | 3 |
| PLAIN-2039 | saliva | 0.361905 | +0.003933 | 0.069431 | 2 |
| PLAIN-487 | addiction | 0.426660 | +0.005034 | 0.921602 | 11 |
| PLAIN-1567 | MAO Inhibitors | 0.385253 | +0.010301 | 0.591794 | 5 |
| PLAIN-1750 | oils | 0.367905 | +0.012081 | 0.164220 | 4 |
| PLAIN-2589 | Anti-inflammatory Life is a Bowl of Cherries | 0.380219 | +0.012920 | 0.857205 | 13 |
| PLAIN-319 | Bowel movements: the scoop on poop | 0.398351 | +0.015712 | 0.069431 | 1 |
| PLAIN-1644 | mitochondria | 0.358897 | +0.021424 | 0.293456 | 3 |
| PLAIN-55 | Foods for Macular Degeneration | 0.370236 | +0.028817 | 0.797517 | 8 |
| PLAIN-2519 | How Much Added Sugar Is Too Much? | 0.368019 | +0.028835 | 0.495094 | 5 |
| PLAIN-3190 | Obesity-Causing Pollutants in Food | 0.378206 | +0.030853 | 0.330138 | 5 |
| PLAIN-1689 | nasal irrigation | 0.371443 | +0.032532 | 0.967468 | 3 |
| PLAIN-258 | Nuts Don't Cause Expected Weight Gain | 0.377166 | +0.043374 | 0.430897 | 4 |
| PLAIN-2599 | Alpha Gal and the Lone Star Tick | 0.467618 | +0.063621 | 0.936379 | 13 |
| PLAIN-2679 | Phosphate Additives in Meat Purge and Cola | 0.427922 | +0.066254 | 0.864315 | 15 |
| PLAIN-1526 | liver cancer | 0.355958 | +0.068753 | 0.563788 | 6 |
| PLAIN-1907 | porcine endogenous retroviruses | 0.458550 | +0.069130 | 0.681206 | 7 |
| PLAIN-77 | Phosphate Additives in Chicken Banned Elsewhere | 0.398141 | +0.070730 | 0.793584 | 9 |
| PLAIN-1928 | premenstrual syndrome | 0.396059 | +0.073364 | 0.727330 | 8 |
| PLAIN-816 | candy | 0.389987 | +0.073364 | 0.305235 | 3 |
| PLAIN-1202 | folate | 0.405081 | +0.075696 | 0.393758 | 4 |
| PLAIN-3280 | Brown Rice vs. Black Rice | 0.367144 | +0.091625 | 0.173428 | 3 |
| PLAIN-1330 | health food stores | 0.368836 | +0.094788 | 0.469000 | 4 |
| PLAIN-1836 | persistent organic pollutants | 0.379628 | +0.094788 | 0.685120 | 11 |
| PLAIN-3451 | Blue-Green Algae & Spirulina | 0.383336 | +0.107789 | 0.876016 | 6 |
| PLAIN-2479 | Stool pH and Colon Cancer | 0.412747 | +0.112679 | 0.312529 | 4 |
| PLAIN-1235 | galactose | 0.419889 | +0.131205 | 0.000000 | 1 |
| PLAIN-680 | beta-carboline alkaloids | 0.357095 | +0.136603 | 0.660159 | 7 |
| PLAIN-1846 | phenolics | 0.383882 | +0.144850 | 0.217261 | 3 |
| PLAIN-3371 | A Harmless Artificial Sweetener | 0.430249 | +0.145626 | 0.217261 | 3 |
| PLAIN-3160 | Egg Cholesterol in the Diet | 0.494726 | +0.150661 | 0.164220 | 3 |
| PLAIN-945 | colon health | 0.394029 | +0.165820 | 0.536810 | 13 |
| PLAIN-417 | Does wild rice contain the same levels of arsenic as brown rice? | 0.467406 | +0.169580 | 0.529635 | 5 |
| PLAIN-2112 | sodium bicarbonate | 0.355353 | +0.200253 | 0.430677 | 1 |
| PLAIN-1804 | parasites | 0.371356 | +0.204478 | 0.637113 | 11 |
| PLAIN-2196 | sweet potatoes | 0.366791 | +0.218195 | 0.781805 | 13 |
| PLAIN-837 | carotid arteries | 0.356461 | +0.304467 | 0.000000 | 2 |
| PLAIN-2889 | Extra Virgin Olive Oil vs. Nuts | 0.483378 | +0.500000 | 0.500000 | 1 |

### Disagreement-only selections

| Query ID | Query | Predicted score | Actual gain | Base NDCG | Judged positives in pool |
|---|---|---:|---:|---:|---:|
| PLAIN-3341 | Are Avocados Good for You? | 0.332356 | +0.411371 | 0.174371 | 3 |
| PLAIN-2091 | sinusitis | 0.266628 | +0.403836 | 0.543067 | 3 |
| PLAIN-2229 | thermogenics | 0.262685 | +0.312852 | 0.156426 | 1 |
| PLAIN-2208 | tapeworms | 0.265397 | +0.310493 | 0.224761 | 4 |
| PLAIN-1578 | mastectomy | 0.309888 | +0.289043 | 0.422575 | 6 |
| PLAIN-1108 | endocannabinoids | 0.302523 | +0.287737 | 0.416181 | 2 |
| PLAIN-1971 | quercitin | 0.240372 | +0.273811 | 0.085143 | 2 |
| PLAIN-3270 | Flaxseeds vs. Chia Seeds | 0.293729 | +0.273587 | 0.000000 | 4 |
| PLAIN-3130 | Are Avocados Bad for You? | 0.338170 | +0.235349 | 0.094788 | 2 |
| PLAIN-1620 | methylmalonic acid | 0.323025 | +0.221282 | 0.383649 | 2 |
| PLAIN-2569 | Prevent Cancer From Going on TOR | 0.259408 | +0.179957 | 0.312049 | 4 |
| PLAIN-1515 | limonin | 0.206284 | +0.172279 | 0.130324 | 1 |
| PLAIN-622 | Ativan | 0.249771 | +0.151301 | 0.000000 | 1 |
| PLAIN-644 | bad breath | 0.290912 | +0.126471 | 0.126471 | 1 |
| PLAIN-269 | Preventing and Treating Kidney Failure With Diet | 0.350155 | +0.125303 | 0.094788 | 1 |
| PLAIN-912 | cilantro | 0.259028 | +0.117516 | 0.636682 | 3 |
| PLAIN-1007 | DDT | 0.255550 | +0.110066 | 0.208273 | 6 |
| PLAIN-1298 | grilling | 0.255627 | +0.110046 | 0.220092 | 2 |
| PLAIN-770 | Broccomax | 0.280588 | +0.106337 | 0.000000 | 1 |
| PLAIN-2489 | Why Prevention is Worth a Ton of Cure | 0.293391 | +0.091092 | 0.000000 | 1 |
| PLAIN-965 | Cornell | 0.224005 | +0.086714 | 0.086714 | 1 |
| PLAIN-2144 | SSRI's | 0.232985 | +0.073548 | 0.484712 | 8 |
| PLAIN-2060 | sea vegetables | 0.336162 | +0.073364 | 0.000000 | 1 |
| PLAIN-2689 | Enhanced Athletic Recovery Without Undermining Adaptation | 0.236984 | +0.063719 | 0.633220 | 8 |
| PLAIN-1440 | jam | 0.168653 | +0.053719 | 0.085143 | 1 |
| PLAIN-1856 | phytates | 0.286314 | +0.053496 | 0.469000 | 4 |
| PLAIN-1484 | lamb | 0.281790 | +0.044074 | 0.094788 | 1 |
| PLAIN-206 | Foods That May Block Cancer Formation | 0.338198 | +0.043792 | 0.066254 | 1 |
| PLAIN-2395 | xylitol | 0.183226 | +0.043792 | 0.286346 | 3 |
| PLAIN-1536 | lovastatin | 0.215438 | +0.015258 | 0.094788 | 1 |
| PLAIN-1949 | Prozac | 0.294215 | +0.014777 | 0.063621 | 6 |
| PLAIN-1386 | Hydroxycut | 0.255268 | +0.014711 | 0.539929 | 3 |
| PLAIN-1065 | Dr. Walter Kempner | 0.314400 | +0.012600 | 0.222446 | 2 |
| PLAIN-1049 | Dr. David Spence | 0.340898 | +0.000000 | 0.000000 | 0 |
| PLAIN-1077 | dulse | 0.208389 | +0.000000 | 0.000000 | 0 |
| PLAIN-1097 | Eggbeaters | 0.270327 | +0.000000 | 0.000000 | 0 |
| PLAIN-1171 | fennel | 0.216112 | +0.000000 | 0.000000 | 0 |
| PLAIN-1181 | figs | 0.153816 | +0.000000 | 0.252943 | 1 |
| PLAIN-1224 | fructans | 0.146345 | +0.000000 | 0.000000 | 0 |
| PLAIN-1273 | goiter | 0.175829 | +0.000000 | 0.000000 | 0 |
| PLAIN-132 | Can We Fight the Blues With Greens? | 0.160548 | +0.000000 | 0.000000 | 0 |
| PLAIN-1362 | hijiki | 0.169079 | +0.000000 | 0.220092 | 1 |
| PLAIN-1495 | lecithin | 0.229933 | +0.000000 | 0.000000 | 0 |
| PLAIN-1505 | lettuce | 0.333092 | +0.000000 | 0.220092 | 2 |
| PLAIN-1556 | maggots | 0.208034 | +0.000000 | 0.000000 | 0 |
| PLAIN-1600 | MelQx | 0.228884 | +0.000000 | 0.000000 | 0 |
| PLAIN-164 | Down But Not Out | 0.179448 | +0.000000 | 0.000000 | 0 |
| PLAIN-1678 | mycotoxins | 0.280601 | +0.000000 | 0.000000 | 0 |
| PLAIN-1709 | Neu5Gc | 0.137693 | +0.000000 | 0.220092 | 1 |
| PLAIN-1720 | nightshades | 0.231195 | +0.000000 | 0.000000 | 0 |
| PLAIN-1826 | penis size | 0.254950 | +0.000000 | 0.220092 | 1 |
| PLAIN-185 | The Best Nutrition Bar | 0.323158 | +0.000000 | 0.220092 | 1 |
| PLAIN-1886 | pneumovax | 0.236936 | +0.000000 | 0.000000 | 0 |
| PLAIN-2018 | Richard Dawkins | 0.156831 | +0.000000 | 0.000000 | 0 |
| PLAIN-2029 | rutabagas | 0.223468 | +0.000000 | 0.000000 | 0 |
| PLAIN-2300 | uterine cancer | 0.313578 | +0.000000 | 0.000000 | 0 |
| PLAIN-2310 | Vaseline | 0.249201 | +0.000000 | 0.000000 | 0 |
| PLAIN-2320 | veggie burgers | 0.241892 | +0.000000 | 0.000000 | 0 |
| PLAIN-2331 | Vioxx | 0.220019 | +0.000000 | 0.235046 | 1 |
| PLAIN-2363 | weeds | 0.249704 | +0.000000 | 0.000000 | 0 |
| PLAIN-2374 | Whitecoat project | 0.252377 | +0.000000 | 0.000000 | 0 |
| PLAIN-2439 | From Table to Able: Combating Disabling Diseases with Food | 0.269420 | +0.000000 | 0.000000 | 0 |
| PLAIN-2449 | How to Make Your Own Fruit and Vegetable Wash | 0.272769 | +0.000000 | 0.000000 | 0 |
| PLAIN-2729 | Fighting the Blues With Greens? | 0.252066 | +0.000000 | 0.000000 | 0 |
| PLAIN-2769 | The McGovern Report | 0.227601 | +0.000000 | 0.000000 | 0 |
| PLAIN-2829 | Unsweetening the Diet | 0.320499 | +0.000000 | 0.000000 | 1 |
| PLAIN-2969 | SAD States: Standard American Diet State-By-State Comparison | 0.293932 | +0.000000 | 0.000000 | 0 |
| PLAIN-3036 | Asparagus Pee | 0.242430 | +0.000000 | 0.000000 | 0 |
| PLAIN-3052 | Anti Up on the Veggies | 0.240296 | +0.000000 | 0.000000 | 0 |
| PLAIN-3180 | Is Kimchi Good For You? | 0.312615 | +0.000000 | 0.000000 | 0 |
| PLAIN-331 | How to live longer in four easy steps | 0.240519 | +0.000000 | 0.000000 | 0 |
| PLAIN-3401 | Superfood Bargains | 0.228245 | +0.000000 | 0.000000 | 0 |
| PLAIN-370 | Do microgreens have more nutrition? | 0.228295 | +0.000000 | 0.302602 | 1 |
| PLAIN-394 | Should we take algae supplements for astaxanthin? | 0.248735 | +0.000000 | 0.000000 | 0 |
| PLAIN-477 | acarbose | 0.229912 | +0.000000 | 0.000000 | 0 |
| PLAIN-592 | apigen | 0.233171 | +0.000000 | 0.000000 | 0 |
| PLAIN-602 | Aricept | 0.228182 | +0.000000 | 0.000000 | 0 |
| PLAIN-710 | blood cancer | 0.274742 | +0.000000 | 0.000000 | 1 |
| PLAIN-720 | bluefish | 0.192509 | +0.000000 | 0.000000 | 0 |
| PLAIN-781 | burritos | 0.255999 | +0.000000 | 0.000000 | 0 |
| PLAIN-901 | chlorination | 0.272732 | +0.000000 | 0.000000 | 0 |
| PLAIN-976 | cretinism | 0.197801 | +0.000000 | 0.390380 | 1 |
| PLAIN-986 | cucumbers | 0.261704 | +0.000000 | 0.000000 | 0 |
| PLAIN-996 | cytoskeleton | 0.181818 | +0.000000 | 0.000000 | 0 |
| PLAIN-1783 | oxalates | 0.207695 | -0.044074 | 0.358954 | 2 |
| PLAIN-1610 | mesothelioma | 0.254176 | -0.053719 | 0.358954 | 2 |
| PLAIN-2619 | Fully Consensual Heart Disease Treatment | 0.259420 | -0.063621 | 0.142019 | 4 |
| PLAIN-3150 | Golden Glow | 0.279950 | -0.069431 | 0.133052 | 3 |
| PLAIN-11 | Living Longer by Reducing Leucine Intake | 0.315196 | -0.125303 | 0.358954 | 5 |
| PLAIN-570 | animal studies | 0.314017 | -0.138862 | 0.138862 | 1 |
| PLAIN-1982 | ranking foods | 0.323823 | -0.183312 | 0.268455 | 3 |
| PLAIN-1213 | formaldehyde | 0.244040 | -0.218407 | 0.218407 | 1 |
| PLAIN-1816 | peaches | 0.293068 | -0.248908 | 0.248908 | 2 |

## regression_tree

Selected outcomes: 33 positive, 48 zero, 16 negative.

Unique predicted scores: 8. Budget-boundary score has 219 tied queries, of which 5 are selected by query-ID order. Learned-only selections: 37, total realized gain +0.513375; disagreement-only selections: 37, total realized gain +1.371691. Divide the gain-sum difference by 324 for the policy NDCG difference.

### Highest predicted scores (20 queries)

| Query ID | Query | Predicted score | Actual gain | Base NDCG | Judged positives in pool |
|---|---|---:|---:|---:|---:|
| PLAIN-1108 | endocannabinoids | 0.104500 | +0.287737 | 0.416181 | 2 |
| PLAIN-1515 | limonin | 0.104500 | +0.172279 | 0.130324 | 1 |
| PLAIN-1678 | mycotoxins | 0.104500 | +0.000000 | 0.000000 | 0 |
| PLAIN-1949 | Prozac | 0.104500 | +0.014777 | 0.063621 | 6 |
| PLAIN-2208 | tapeworms | 0.104500 | +0.310493 | 0.224761 | 4 |
| PLAIN-901 | chlorination | 0.104500 | +0.000000 | 0.000000 | 0 |
| PLAIN-2429 | Diverticulosis: When Our Most Common Gut Disorder Hardly Existed | 0.090683 | +0.018642 | 0.453904 | 3 |
| PLAIN-2969 | SAD States: Standard American Diet State-By-State Comparison | 0.090683 | +0.000000 | 0.000000 | 0 |
| PLAIN-304 | EPA dioxin limit has National Chicken Council worried products could be declared "unfit for consumption" | 0.090683 | +0.000000 | 0.000000 | 0 |
| PLAIN-1087 | Easter Island | 0.049584 | +0.133817 | 0.101229 | 1 |
| PLAIN-1546 | lycopene | 0.049584 | -0.025001 | 0.620762 | 7 |
| PLAIN-164 | Down But Not Out | 0.049584 | +0.000000 | 0.000000 | 0 |
| PLAIN-2374 | Whitecoat project | 0.049584 | +0.000000 | 0.000000 | 0 |
| PLAIN-2769 | The McGovern Report | 0.049584 | +0.000000 | 0.000000 | 0 |
| PLAIN-508 | air travel | 0.049584 | +0.000000 | 0.000000 | 0 |
| PLAIN-1017 | detoxification | 0.023103 | -0.110046 | 0.220092 | 3 |
| PLAIN-1213 | formaldehyde | 0.023103 | -0.218407 | 0.218407 | 1 |
| PLAIN-1235 | galactose | 0.023103 | +0.131205 | 0.000000 | 1 |
| PLAIN-1261 | glaucoma | 0.023103 | -0.016390 | 0.648932 | 5 |
| PLAIN-1373 | hormesis | 0.023103 | -0.015258 | 0.469000 | 3 |

### Most harmful selected queries

| Query ID | Query | Predicted score | Actual gain | Base NDCG | Judged positives in pool |
|---|---|---:|---:|---:|---:|
| PLAIN-1816 | peaches | 0.012832 | -0.248908 | 0.248908 | 2 |
| PLAIN-1213 | formaldehyde | 0.023103 | -0.218407 | 0.218407 | 1 |
| PLAIN-2270 | tryptophan | 0.023103 | -0.150431 | 0.654809 | 3 |
| PLAIN-1017 | detoxification | 0.023103 | -0.110046 | 0.220092 | 3 |
| PLAIN-3150 | Golden Glow | 0.012832 | -0.069431 | 0.133052 | 3 |
| PLAIN-2080 | sheep | 0.012832 | -0.061643 | 0.449887 | 5 |
| PLAIN-670 | bell peppers | 0.012832 | -0.058106 | 0.168152 | 2 |
| PLAIN-1610 | mesothelioma | 0.023103 | -0.053719 | 0.358954 | 2 |
| PLAIN-1783 | oxalates | 0.023103 | -0.044074 | 0.358954 | 2 |
| PLAIN-2407 | zingerone | 0.012832 | -0.028817 | 0.358954 | 2 |
| PLAIN-955 | cookies | 0.012832 | -0.028817 | 0.358954 | 2 |
| PLAIN-1546 | lycopene | 0.049584 | -0.025001 | 0.620762 | 7 |
| PLAIN-1261 | glaucoma | 0.023103 | -0.016390 | 0.648932 | 5 |
| PLAIN-1373 | hormesis | 0.023103 | -0.015258 | 0.469000 | 3 |
| PLAIN-1038 | dogs | -0.004239 | -0.009447 | 0.769526 | 8 |
| PLAIN-1027 | dietary guidelines | -0.004239 | -0.000678 | 0.586795 | 8 |
| PLAIN-1 | Why Deep Fried Foods May Cause Cancer | -0.004239 | +0.000000 | 0.358954 | 2 |
| PLAIN-101 | How to Treat Multiple Sclerosis With Diet | -0.004239 | +0.000000 | 0.563788 | 5 |
| PLAIN-1049 | Dr. David Spence | -0.004239 | +0.000000 | 0.000000 | 0 |
| PLAIN-1077 | dulse | 0.012832 | +0.000000 | 0.000000 | 0 |

### Largest missed gains

| Query ID | Query | Predicted score | Actual gain | Base NDCG | Judged positives in pool |
|---|---|---:|---:|---:|---:|
| PLAIN-2889 | Extra Virgin Olive Oil vs. Nuts | -0.004239 | +0.500000 | 0.500000 | 1 |
| PLAIN-3084 | How the Institute of Medicine Arrived at Their Vitamin D Recommendation | -0.016859 | +0.500000 | 0.000000 | 1 |
| PLAIN-3341 | Are Avocados Good for You? | -0.004239 | +0.411371 | 0.174371 | 3 |
| PLAIN-837 | carotid arteries | -0.004239 | +0.304467 | 0.000000 | 2 |
| PLAIN-3270 | Flaxseeds vs. Chia Seeds | -0.004239 | +0.273587 | 0.000000 | 4 |
| PLAIN-3130 | Are Avocados Bad for You? | -0.004239 | +0.235349 | 0.094788 | 2 |
| PLAIN-152 | How Probiotics Affect Mental Health | -0.004239 | +0.230474 | 0.769526 | 13 |
| PLAIN-2979 | Cow’s Milk Casomorphin and Crib Death | -0.004239 | +0.226294 | 0.386853 | 1 |
| PLAIN-1620 | methylmalonic acid | -0.004239 | +0.221282 | 0.383649 | 2 |
| PLAIN-2196 | sweet potatoes | -0.004239 | +0.218195 | 0.781805 | 13 |
| PLAIN-3210 | Caffeine During Pregnancy | -0.004239 | +0.201657 | 0.438287 | 9 |
| PLAIN-2112 | sodium bicarbonate | -0.004239 | +0.200253 | 0.430677 | 1 |
| PLAIN-690 | bio identical hormones | -0.004239 | +0.194813 | 0.107789 | 1 |
| PLAIN-1740 | nutritional yeast | -0.004239 | +0.182654 | 0.371489 | 4 |
| PLAIN-2990 | Watermelon as Treatment for Erectile Dysfunction | -0.004239 | +0.180091 | 0.139056 | 2 |
| PLAIN-2569 | Prevent Cancer From Going on TOR | -0.004239 | +0.179957 | 0.312049 | 4 |
| PLAIN-417 | Does wild rice contain the same levels of arsenic as brown rice? | -0.016859 | +0.169580 | 0.529635 | 5 |
| PLAIN-2809 | Apple Skin: Peeling Back Cancer | -0.004239 | +0.166596 | 0.397192 | 4 |
| PLAIN-945 | colon health | -0.004239 | +0.165820 | 0.536810 | 13 |
| PLAIN-2459 | How to Prevent a Stroke | -0.004239 | +0.161985 | 0.385413 | 5 |

### Learned-only selections

| Query ID | Query | Predicted score | Actual gain | Base NDCG | Judged positives in pool |
|---|---|---:|---:|---:|---:|
| PLAIN-2270 | tryptophan | 0.023103 | -0.150431 | 0.654809 | 3 |
| PLAIN-2080 | sheep | 0.012832 | -0.061643 | 0.449887 | 5 |
| PLAIN-670 | bell peppers | 0.012832 | -0.058106 | 0.168152 | 2 |
| PLAIN-2407 | zingerone | 0.012832 | -0.028817 | 0.358954 | 2 |
| PLAIN-955 | cookies | 0.012832 | -0.028817 | 0.358954 | 2 |
| PLAIN-1546 | lycopene | 0.049584 | -0.025001 | 0.620762 | 7 |
| PLAIN-1261 | glaucoma | 0.023103 | -0.016390 | 0.648932 | 5 |
| PLAIN-1373 | hormesis | 0.023103 | -0.015258 | 0.469000 | 3 |
| PLAIN-1038 | dogs | -0.004239 | -0.009447 | 0.769526 | 8 |
| PLAIN-1027 | dietary guidelines | -0.004239 | -0.000678 | 0.586795 | 8 |
| PLAIN-1 | Why Deep Fried Foods May Cause Cancer | -0.004239 | +0.000000 | 0.358954 | 2 |
| PLAIN-101 | How to Treat Multiple Sclerosis With Diet | -0.004239 | +0.000000 | 0.563788 | 5 |
| PLAIN-1397 | IgA | 0.012832 | +0.000000 | 0.000000 | 0 |
| PLAIN-1418 | insecticides | 0.023103 | +0.000000 | 0.000000 | 0 |
| PLAIN-2050 | saturate fat | 0.023103 | +0.000000 | 0.000000 | 0 |
| PLAIN-2123 | Spain | 0.012832 | +0.000000 | 0.000000 | 0 |
| PLAIN-2133 | spirulina | 0.023103 | +0.000000 | 0.693022 | 5 |
| PLAIN-2219 | telomeres | 0.023103 | +0.000000 | 1.000000 | 16 |
| PLAIN-304 | EPA dioxin limit has National Chicken Council worried products could be declared "unfit for consumption" | 0.090683 | +0.000000 | 0.000000 | 0 |
| PLAIN-3073 | Glycotoxins | 0.023103 | +0.000000 | 0.237198 | 1 |
| PLAIN-508 | air travel | 0.049584 | +0.000000 | 0.000000 | 0 |
| PLAIN-560 | anesthesia | 0.023103 | +0.000000 | 0.000000 | 0 |
| PLAIN-582 | antihistamines | 0.023103 | +0.000000 | 0.000000 | 0 |
| PLAIN-612 | asbestos | 0.023103 | +0.000000 | 0.000000 | 0 |
| PLAIN-740 | BPA | 0.023103 | +0.000000 | 0.804810 | 3 |
| PLAIN-859 | cellulite | 0.023103 | +0.000000 | 0.539003 | 3 |
| PLAIN-2429 | Diverticulosis: When Our Most Common Gut Disorder Hardly Existed | 0.090683 | +0.018642 | 0.453904 | 3 |
| PLAIN-1644 | mitochondria | 0.023103 | +0.021424 | 0.293456 | 3 |
| PLAIN-2249 | tongue cancer | 0.023103 | +0.028817 | 0.330138 | 2 |
| PLAIN-1771 | organochlorines | 0.023103 | +0.032749 | 0.330138 | 3 |
| PLAIN-550 | ammonia | 0.023103 | +0.051197 | 0.159589 | 2 |
| PLAIN-791 | cabbage | 0.023103 | +0.057330 | 0.403483 | 6 |
| PLAIN-816 | candy | 0.023103 | +0.073364 | 0.305235 | 3 |
| PLAIN-1235 | galactose | 0.023103 | +0.131205 | 0.000000 | 1 |
| PLAIN-1087 | Easter Island | 0.049584 | +0.133817 | 0.101229 | 1 |
| PLAIN-848 | cats | 0.012832 | +0.154939 | 0.535709 | 6 |
| PLAIN-1804 | parasites | 0.023103 | +0.204478 | 0.637113 | 11 |

### Disagreement-only selections

| Query ID | Query | Predicted score | Actual gain | Base NDCG | Judged positives in pool |
|---|---|---:|---:|---:|---:|
| PLAIN-3341 | Are Avocados Good for You? | -0.004239 | +0.411371 | 0.174371 | 3 |
| PLAIN-3270 | Flaxseeds vs. Chia Seeds | -0.004239 | +0.273587 | 0.000000 | 4 |
| PLAIN-3130 | Are Avocados Bad for You? | -0.004239 | +0.235349 | 0.094788 | 2 |
| PLAIN-1620 | methylmalonic acid | -0.004239 | +0.221282 | 0.383649 | 2 |
| PLAIN-2569 | Prevent Cancer From Going on TOR | -0.004239 | +0.179957 | 0.312049 | 4 |
| PLAIN-644 | bad breath | -0.004239 | +0.126471 | 0.126471 | 1 |
| PLAIN-269 | Preventing and Treating Kidney Failure With Diet | -0.004239 | +0.125303 | 0.094788 | 1 |
| PLAIN-1655 | mortality | -0.004239 | +0.106113 | 0.641046 | 15 |
| PLAIN-2489 | Why Prevention is Worth a Ton of Cure | -0.004239 | +0.091092 | 0.000000 | 1 |
| PLAIN-2060 | sea vegetables | -0.004239 | +0.073364 | 0.000000 | 1 |
| PLAIN-2689 | Enhanced Athletic Recovery Without Undermining Adaptation | -0.004239 | +0.063719 | 0.633220 | 8 |
| PLAIN-206 | Foods That May Block Cancer Formation | -0.004239 | +0.043792 | 0.066254 | 1 |
| PLAIN-1065 | Dr. Walter Kempner | -0.004239 | +0.012600 | 0.222446 | 2 |
| PLAIN-1181 | figs | -0.042451 | +0.000000 | 0.252943 | 1 |
| PLAIN-1224 | fructans | -0.042451 | +0.000000 | 0.000000 | 0 |
| PLAIN-132 | Can We Fight the Blues With Greens? | -0.004239 | +0.000000 | 0.000000 | 0 |
| PLAIN-1362 | hijiki | -0.042451 | +0.000000 | 0.220092 | 1 |
| PLAIN-1709 | Neu5Gc | -0.042451 | +0.000000 | 0.220092 | 1 |
| PLAIN-1826 | penis size | -0.004239 | +0.000000 | 0.220092 | 1 |
| PLAIN-185 | The Best Nutrition Bar | -0.004239 | +0.000000 | 0.220092 | 1 |
| PLAIN-2018 | Richard Dawkins | -0.042451 | +0.000000 | 0.000000 | 0 |
| PLAIN-2439 | From Table to Able: Combating Disabling Diseases with Food | -0.004239 | +0.000000 | 0.000000 | 0 |
| PLAIN-2449 | How to Make Your Own Fruit and Vegetable Wash | -0.004239 | +0.000000 | 0.000000 | 0 |
| PLAIN-2729 | Fighting the Blues With Greens? | -0.004239 | +0.000000 | 0.000000 | 0 |
| PLAIN-2829 | Unsweetening the Diet | -0.004239 | +0.000000 | 0.000000 | 1 |
| PLAIN-3052 | Anti Up on the Veggies | -0.004239 | +0.000000 | 0.000000 | 0 |
| PLAIN-3180 | Is Kimchi Good For You? | -0.004239 | +0.000000 | 0.000000 | 0 |
| PLAIN-331 | How to live longer in four easy steps | -0.004239 | +0.000000 | 0.000000 | 0 |
| PLAIN-370 | Do microgreens have more nutrition? | -0.004239 | +0.000000 | 0.302602 | 1 |
| PLAIN-394 | Should we take algae supplements for astaxanthin? | -0.004239 | +0.000000 | 0.000000 | 0 |
| PLAIN-710 | blood cancer | -0.004239 | +0.000000 | 0.000000 | 1 |
| PLAIN-976 | cretinism | -0.042451 | +0.000000 | 0.390380 | 1 |
| PLAIN-2619 | Fully Consensual Heart Disease Treatment | -0.004239 | -0.063621 | 0.142019 | 4 |
| PLAIN-2385 | World Health Organization | -0.004239 | -0.081210 | 0.154574 | 3 |
| PLAIN-11 | Living Longer by Reducing Leucine Intake | -0.004239 | -0.125303 | 0.358954 | 5 |
| PLAIN-570 | animal studies | -0.004239 | -0.138862 | 0.138862 | 1 |
| PLAIN-1982 | ranking foods | -0.004239 | -0.183312 | 0.268455 | 3 |
