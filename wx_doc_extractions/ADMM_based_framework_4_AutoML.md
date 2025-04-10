# An ADMM Based Framework for AutoML Pipeline Configuration

# Sijia Liu,* Parikshit Ram,* Deepak Vijaykeerthy, Djallel Bouneffouf, Gregory Bramble, Horst Samulowitz, Dakuo Wang, Andrew Conn, Alexander Gray

IBM Research AI

*

Equal contributions

## Abstract

We study the AutoML problem of automatically configuring machine learning pipelines by jointly selecting algorithms and their appropriate hyper-parameters for all steps in supervised learning pipelines. This black-box (gradient-free) optimization with mixed integer & continuous variables is a challenging problem. We propose a novel AutoML scheme by leveraging the alternating direction method of multipliers (ADMM). The proposed framework is able to (i) decompose the optimization problem into easier sub-problems that have a reduced number of variables and circumvent the challenge of mixed variable categories, and (ii) incorporate black-box constraints along side the black-box optimization objective. We empirically evaluate the flexibility (in utilizing existing AutoML tech niques), effectiveness (against open source AutoML toolkits), and unique capability (of executing AutoML with practically motivated black-box constraints) of our proposed scheme on a collection of binary classification data sets from UCI ML & OpenML repositories. We observe that on an average our framework provides significant gains in comparison to other AutoML frameworks (Auto-sklearn & TPOT), highlighting the practical advantages of this framework.

## 1 Introduction

Automated machine learning (AutoML) research has re ceived increasing attention. The focus has shifted from hyper parameter optimization (HPO) for the best configuration of a single machine learning (ML) algorithm (Snoek, Larochelle, and Adams 2012), to configuring multiple stages of a ML pipeline (e.g., transformations, feature selection, predictive modeling) (Feurer et al. 2015). Among the wide-range of research challenges offered by AutoML, we focus on the automatic pipeline configuration problem (that is, joint algo rithm selection and HPO), and tackle it from the perspective of mixed continuous-integer black-box nonlinear program ming. This problem has two main challenges: (i) the tight coupling between the ML algorithm selection & HPO; and (ii) the black-box nature of optimization objective lacking any explicit functional form and gradients – optimization feedback is only available in the form of function evalua tions. We propose a new AutoML framework to address these challenges by leveraging the alternating direction method of multipliers (ADMM). ADMM offers a two-block alternating optimization procedure that splits an involved problem (with

Copyright

c

2020, Association for the Advancement of Artificial

Intelligence (www.aaai.org). All rights reserved.

multiple variables & constraints) into simpler sub-problems (Boyd and others 2011)

Contributions. Starting with a combinatorially large set of algorithm candidates and their collective set of hyper parameters, we utilize ADMM to decompose the AutoML problem into three problems: (i) HPO with a small set of only continuous variables & constraints, (ii) a closed-form Euclidean projection onto an integer set, and (iii) a combi natorial problem of algorithm selection. Moreover, we ex ploit the ADMM framework to handle any black-box con straints alongside the black-box objective (loss) function – the above decomposition seamlessly incorporates such con straints while retaining almost the same sub-problems.

Our contributions are: (i) We explicitly model the coupling between hyper-parameters and available algorithms, and ex ploit the hidden structure in the AutoML problem (Section 3). (ii) We employ ADMM to decompose the problem into a sequence of sub-problems (Section 4.1), which decouple the difficulties in AutoML and can each be solved more effi ciently and effectively, demonstrating over 10× speedup and 10% improvement in many cases. (iii) We present the first AutoML framework that explicitly handles general black-box constraints (Section 4.2). (iv) We demonstrate the flexibility and effectiveness of the ADMM-based scheme empirically against popular AutoML toolkits Auto-sklearn (Feurer et al. 2015) & TPOT (Olson and Moore 2016) (Section 5), per forming best on 50% of the datasets; Auto-sklearn performed best on 27% and TPOT on 20%.

## 2 Related work

Black-box optimization in AutoML. Beyond grid-search for HPO, random search is a very competitive baseline be cause of its simplicity and parallelizability (Bergstra and Bengio 2012). Sequential model-based optimization (SMBO) (Hutter, Hoos, and Leyton-Brown 2011) is a common tech nique with different ‘models’ such as Gaussian processes (Snoek, Larochelle, and Adams 2012), random forests (Hut ter, Hoos, and Leyton-Brown 2011) and tree-parzen estima tors (Bergstra et al. 2011). However, black-box optimization is a time consuming process because the expensive black box function evaluation involves model training and scoring (on a held-out set). Efficient multi-fidelity approximations of the black-box function based on some budget (training samples/epochs) combined with bandit learning can skip un promising candidates early via successive halving (Jamieson and Talwalkar 2016; Sabharwal and others 2016) and Hy perBand (Li et al. 2018). However, these schemes essen tially perform an efficient random search and are well suited for search over discrete spaces or discretized continuous spaces. BOHB (Falkner, Klein, and Hutter 2018) combines SMBO (with TPE) and HyperBand for improved optimiza tion. Meta-learning (Vanschoren 2018) leverages past expe riences in the optimization with search space refinements and promising starting points. The collaborative filtering based methods (Yang et al. 2019) are examples of meta learning, where information from past evaluation on other datasets is utilized to pick pipelines for any new datasets. Compared to the recent works on iterative pipeline construc tion using tree search (Mohr, Wever, and Hüllermeier 2018; Rakotoarison, Schoenauer, and Sebag 2019), we provide a natural yet formal primal-dual decomposition of autoML pipeline configuration problems.

Toolkits. Auto-WEKA (Thornton et al. 2012; Kotthoff and others 2017) and Auto-sklearn (Feurer et al. 2015) are the main representatives of SBMO-based AutoML. Both apply the general purpose framework SMAC (Sequential Model based Algorithm Configuration) (Hutter, Hoos, and Leyton Brown 2011) to find optimal ML pipelines. Both consider a fixed shape of the pipeline with functional modules (pre processing, transforming, modeling) and automatically select a ML algorithm and its hyper-parameters for each module. Auto-sklearn improves upon Auto-WEKA with two inno vations: (i) a meta-learning based preprocessing step that uses ‘meta-features’ of the dataset to determine good initial pipeline candidates based on past experience to warm start the optimization, (ii) an greedy forward-selection ensembling (Caruana et al. 2004) of the pipeline configurations found dur ing the optimization as an independent post-processing step. Hyperopt-sklearn (Komer, Bergstra, and Eliasmith 2014) uti lizes TPE as the SMBO. TPOT (Olson and Moore 2016) and ML-Plan (Mohr, Wever, and Hüllermeier 2018) use genetic algorithm and hierarchical task networks planning respec tively to optimize over the pipeline shape and the algorithm choices, but require discretization of the hyper-parameter space (which can be inefficient in practice as it leads perfor mance degradation). AlphaD3M (Drori, Krishnamurthy, and others 2018) integrates reinforcement learning with Monte Carlo tree search (MCTS) for solving AutoML problems but without imposing efficient decomposition over hyperpa rameters and model selection. AutoStacker (Chen, Wu, and others 2018) focuses on ensembling and cascading to gener ate complex pipelines and the actual algorithm selection and hyper-parameter optimization happens via random search.

## 3 An Optimization Perspective to AutoML

We focus on the joint algorithm selection and HPO for a fixed pipeline – a ML pipeline with a fixed sequence of functional modules (preprocessing → missing/categorical handling → transformations → feature selection → modeling) with a set of algorithm choices in each module – termed asthe CASH (combined algorithm selection and HPO) problem (Thornton et al. 2012; Feurer et al. 2015) and solved with toolkits such as Auto-WEKA and Auto-sklearn. We extend this formula tion by explicitly expressing the combinatorial nature of the algorithm selection with Boolean variables and constraints.

We will also briefly discuss how this formulation facilities extension to other flexible pipelines.

Problem statement. For N functional modules (e.g., pre processor, transformer, estimator) with a choice of Ki algo Ki rithms in each, let zi ∈ {0, 1} denote the algorithm choice in module i, with the constraint 1 >zi = PKi zij = 1 j=1 ensuring that only a single algorithm is chosen from each module. Let z = {z1, . . . , zN }. Assuming that categorical hyper-parameters can be encoded as integers (using standard techniques), let θij be the hyper-parameters of algorithm j c mc in module i, with θ ∈ Cij ⊂ R ij as the continuous hyper ij

d md parameters (constrained to the set Cij ) and θ ∈ Dij ⊂ Z ij ij as the integer hyper-parameters (constrained to Dij ). Condi tional hyper-parameters can be handled with additional con straints θij ∈ Eij or by “flattening” the hyper-parameter tree and considering each leaf as a different algorithm. For sim plicity of exposition, we assume that the conditional hyper parameters are flattened into additional algorithm choices. Let θ = {θij , ∀i ∈ [N], j ∈ [Ki ]}, where [n] = {1, . . . , n} for n ∈ N. Let f (z, θ; A) represent some notion of loss of a ML pipeline corresponding to the algorithm choices as per z with the hyper-parameters θ on a learning task with data A (such as the k-fold cross-validation or holdout valida tion loss). The optimization problem of automatic pipeline configuration is stated as:

min f(z, θ; A) z,θ Ki (1)  zi ∈ {0, 1} , 1 >zi = 1, ∀i ∈ [N], subject to c d θ ∈ Cij , θ ∈ Dij , ∀i ∈ [N], j ∈ [Ki]. ij ij

We highlight 2 key differences of problem (1) from the conventional CASH formulation: (i) we use explicit Boolean variables z to encode the algorithm selection, (ii) we differ entiate continuous variables/constraints from discrete ones for a possible efficient decomposition between continuous optimization and integer programming. These features better characterize the properties of the problem and thus enable more effective joint optimization. For any given (z, θ) and data A, the objective (loss) function f(z, θ; A) is a black box function – it does not have an analytic form with respect to (z, θ) (hence no derivatives). The actual evaluation of f usually involves training, testing and scoring the ML pipeline corresponding to (z, θ) on some split of the data A.

AutoML with black-box constraints. With the increas ing adoption of AutoML, the formulation (1) may not be sufficient. AutoML may need to find ML pipelines with high predictive performance (low loss) that also explicitly sat isfy application specific constraints. Deployment constraints may require the pipeline to have prediction latency or size in memory below some threshold (latency ≤ 10µs, mem ory ≤ 100MB). Business specific constraints may desire pipelines with low overall classification error and an explicit upper bound on the false positive rate – in a loan default risk application, false positives leads to loan denials to eligible candidates, which may violate regulatory requirements. In the quest for fair AI, regulators may explicitly require the ML pipeline to be above some predefined fairness threshold (Friedler and others 2019). Furthermore, many applications have very domain specific metric(s) with corresponding con straints – custom metrics are common in Kaggle competitions. We incorporate such requirements by extending AutoML for mulation (1) to include M black-box constraints:

gi

(z, θ; A) ≤ i, i ∈ [M]. (2)

These functions have no analytic form with respect to (z, θ), in constrast to the analytic constraints in problem (1). One ap proach is to incorporate these constraints into the black-box objective with a penalty function p, where the new objec tive becomes f + P p(gi , i) or f · Q p(gi , i). However, i i these schemes are very sensitive to the choice of the penalty function and do not guarantee feasible solutions.

Generalization for more flexible pipelines. We can ex tend the problem formulation (1) to enable optimization over the ordering of the functional modules. For example, we can choose between ‘preprocessor → transformer → feature se lector’ OR ‘feature selector → preprocessor → transformer’. The ordering of T ≤ N modules can be optimized by in 2 troducing T Boolean variables o = {oik : i, k ∈ [T]}, where oik = 1 indicates that module i is placed at po sition k. The following constraints are then needed: (i) P oik = 1, ∀i ∈ [T] indicates that module i is placed at k∈[T] a single position, and (ii) P oik = 1∀k ∈ [T] enforces i∈[T] that only one module is placed at position k. These variables can be added to z in problem (1) (z = {z1, . . . , zN , o}). The resulting formulation still obeys the generic form of (1), which as will be evident later, can be efficiently solved by an operator splitting framework like ADMM (Boyd and others 2011).

## 4 ADMM-Based Joint Optimizer

ADMM provides a general effective optimization framework to solve complex problems with mixed variables and multiple constraints (Boyd and others 2011; Liu et al. 2018). We utilize this framework to decompose problem (1) without and with black-box constraints (2) into easier sub-problems.

## 4.1 Efficient operator splitting for AutoML

In what follows, we focus on solving problem (1) with an alytic constraints. The handling of black-box constraints c will be elaborated on in the next section. Denoting θ = c {θ , ∀i ∈ [N], j ∈ [Ki ]} as all the continuous hyper ij d parameters and θ (defined correspondingly) as all the integer hyper-parameters, we re-write problem (1) as:

 n c d o  min f z, θ , θ ; A z,θ={θc,θd} Ki (3)  zi ∈ {0, 1} , 1 >zi = 1, ∀i ∈ [N], subject to c d θ ∈ Cij , θ ∈ Dij , ∀i ∈ [N], j ∈ [Ki]. ij ij

Introduction of continuous surrogate loss. With Deij as the continuous relaxation of the integer space Dij (if Dij includes integers ranging from {l, . . . , u} ⊂ Z, then d Deij = [l, u] ⊂ R), and θe as the continuous surrogates for d θ with θeij ∈ Deij (corresponding to θij ∈ Dij ), we utilize a surrogate loss function fefor problem (3) defined solely over the continuous domain with respect to θ:

d d  n c o fe z, θ , θe ; A  := f  z, n θ c , PD  o  θe ; A , (4) d d

where PD(θe ) = {PDij (θe ij ), ∀i ∈ [N], j ∈ [Ki ]} is the projection of the continuous surrogates onto the integer set. This projection is necessary since the black-box function

is defined (hence can only be evaluated) on the integer sets Dij s. Ergo, problem (3) can be equivalently posed as

d  n c o  min fe z, θ , θe ; A z,θc,θed ,δ  zi ∈ {0, 1} Ki , 1 >zi = 1, ∀i ∈ [N]  θ c ∈ Cij , ∈ , ∀i ∈ [N], j ∈ d (5) ij subject to θe ij Deij [Ki] δij ∈ Dij , ∀i ∈ [N], j ∈ [Ki]  d θe ij = δij , ∀i ∈ [N], j ∈ [Ki],

where the equivalence between problems (3) & (5) is es d tablished by the equality constraint θe = δij ∈ Dij , im ij d d d c plying PDij (θe ) = θe ∈ Dij and fe(z, {θ , θe }; A) = ij ij d c f(z, {θ , θe }; A). The continuous surrogate loss (4) is key

in being able to perform theoretically grounded operator split ting (via ADMM) over mixed continuous/integer variables in the AutoML problem (3).

Operator splitting from ADMM. Using the notation that IX (x) = 0 if x ∈ X else +∞, and defining the sets Z = Ki {z: z = {zi}, zi ∈ {0, 1} , 1 >zi = 1, ∀i ∈ [N]}, C = c c c c {θ : θ = {θ ij}, θ ∈ Cij , ∀i ∈ [N], j ∈ [Ki ]}, D = ij {δ : δ = {δij}, δij ∈ Dij , ∀i ∈ [N], j ∈ [Ki ]} and De = d d d d {θe : θe = {θe ij}, θe ∈ Deij , ∀i ∈ [N], j ∈ [Ki ]}, we can ij re-write problem (5) as

d d  n c o  c min fe z, θ , θe ; A + IZ (z) + IC(θ ) + IDe(θe ) z,θc,θed ,δ

+

ID(δ);

subject to

θe

d

= δ.

(6)

with the corresponding augmented Lagrangian function

d d c  n c o  c L(z, θ , θe , δ,λ) := fe z, θ , θe ; A + IZ (z) + IC(θ ) 2 d >  d  ρ d − δ , (7) + IDe(θe ) + ID(δ) + λ θe − δ + θe 2 2

where λ is the Lagrangian multiplier, and ρ > 0 is a penalty parameter for the augmented term.

ADMM (Boyd and others 2011) alternatively minimizes the augmented Lagrangian function (7) over two blocks of variables, leading to an efficient operator splitting framework for nonlinear programs with nonsmooth objective function and equality constraints. Specifically, ADMM solves problem

c (1) by alternatively minimizing (7) over variables {θ , θe }, and {δ, z}. This can be equivalently converted into 3 sub

d

d c problems over variables {θ , θe }, δ and z, respectively. We refer readers to Algorithm 1 for simplified sub-problems and Appendix 11 for detailed derivation.

The rationale behind the advantage of ADMM is that it decomposes the AutoML problem into sub-problems with smaller number of variables: This is crucial in black-box op timization where convergence is strongly dependent on the number of variables. For example, the number of black-box evaluations needed for critical point convergence is typically 3 O(n ∼ n ) for n variables (Larson, Menickelly, and Wild 2019). In what follows, we show that the easier sub-problems in Algorithm 1 yield great interpretation of the AutoML prob lem (1) and suggest efficient solvers in terms of continuous hyper-parameter optimization, integer projection operation, and combinatorial algorithm selection.

1Appendices are at https://arxiv.org/pdf/1905.00424.pdf

Algorithm 1 Operator splitting from ADMM to solve problem

(5)

2 n c(t+1) d(t+1)o  (t) n c d o  c d d (t) 1 (t) θ , θe = arg min fe z , θ , θe ; A + IC(θ ) + IDe(θe ) + (ρ/2) θe − b , b := δ − λ , (θ-min) 2 ρ θc,θed (t+1) 2 d(t+1) (t) δ = arg min ID(δ) + (ρ/2) ka − δk , a := θe + (1/ρ)λ , (δ-min) 2 δ (t+1)  n c(t+1) d(t+1)o  z = arg min fe z, θ , θe ; A + IZ (z), (z-min) z d

where

(t)

represents the iteration index, and the Lagrangian multipliers

λ

are updated as

λ

(t+1)

= λ

(t)

+

ρ(θe

(t+1)

−

δ

(t+1)).

Solving θ-min. Problem (θ-min) can be rewritten as

d 2  (t) n c o  ρ d min fe z , θ , θe ; A + θe − b 2 2 θc,θed c  θ ∈ Cij (8) ij subject to d ∀i ∈ [N], j ∈ [Ki], θe ij ∈ Deij ,

c where both θ and θe are continuous optimization vari (t) ables. Since the algorithm selection scheme z is fixed for this problem, fe in problem (8) only depends on the hyper parameters of the chosen algorithms – the active set of con

d

c (t) tinuous variables (θ , θe ) where zij = 1. This splits ij ij problem (8) even further into two problems. The inactive set problem reduces to the following for all i ∈ [N], j ∈ [Ki ] such that zij = 0:

d

ρ d d 2 min kθe ij − bijk 2 subject to θe ij ∈ Deij , (9) 2 θed ij

which is solved by a Euclidean projection of bij onto Deij .

For the active set of variables S = {(θ , θe ): θ ∈ ij ij ij Cij , θeij ∈ Deij , zij = 1, ∀i ∈ [N], j ∈ [Ki ]}, problem (8) reduces to the following black-box optimization with only the small active set of continuous variables2

c

d

c

d 2  (t) n c o  ρ d min fe z , θ , θe ; A + θe − b . (10) 2 (θc,θed )∈S 2

The above problem can be solved using Bayesian optimiza tion (Shahriari et al. 2016), direct search (Larson, Menickelly, and Wild 2019), or trust-region based derivative-free opti mization (Conn, Scheinberg, and Vicente 2009).

Solving δ-min. According to the definition of D, problem (δ-min) can be rewritten as

ρ 2 min kδ−ak subject to δij ∈ Dij , ∀i ∈ [N], j ∈ [Ki], (11) 2 δ 2

and solved in closed form by projecting a onto De and then rounding to the nearest integer in D. Solving z-min. Problem (z-min) rewritten as

 n c(t+1) d(t+1)o  min fe z, θ , θe ; A z (12) Ki subject to zi ∈ {0, 1} , 1 >zi = 1, ∀i ∈ [N]

2 For the AutoML problems we consider in our empirical evalu

d c tations, |θ| = |θ | + |θe | ≈ 100 while the largest possible active ij ij set S is less than 15 and typically less than 10.

is a black-box integer program solved exactly with QN Ki i=1

evaluations of fe. However, this is generally not feasible. Beyond random sampling, there are a few ways to lever age existing AutoML schemes: (i) Combinatorial multi armed bandits. – Problem (12) can be interpreted through combinatorial bandits as the selection of the optimal N arms (in this case, algorithms) from PN Ki arms based i=1 on bandit feedback and can be efficiently solved with Thompson sampling (Durand and Gagné 2014) (ii) Multi fidelity approximation of black-box evaluations – Techniques such as successive halving (Jamieson and Talwalkar 2016; Li et al. 2018) or incremental data allocation (Sabharwal and others 2016) can efficiently search over a discrete set of QN Ki candidates. (iii) Genetic algorithms – Genetic pro i=1 gramming can perform this discrete black-box optimization starting from a randomly generated population and building the next generation based on the ‘fitness’ of the pipelines and random ‘mutations’ and ‘crossovers’.

## 4.2 ADMM with black-box constraints

We next consider problem (3) in the presence of black-box constraints (2). Without loss of generality, we assume that i ≥ 0 for i ∈ [M]. By introducing scalars ui ∈ [0, i ], we can reformulate the inequality constraint (2) as the equality constraint together with a box constraint

 n c d o  gi z, θ , θ ; A − i + ui, ui ∈ [0, i], i ∈ [M]. (13)

We then introduce a continuous surrogate black-box functions

gei gi , ∀i ∈ [M] in a similar manner to fe given by (4). for Following the reformulation of (3) that lends itself to the application of ADMM, the version with black-box constraints (13) can be equivalently transformed into

d  n c o  min fe z, θ , θe ; A z,θc,θed ,δ Ki  zi ∈ {0, 1} , 1 >zi = 1, ∀i ∈ [N]  ij Cij , θe ij ∈ Deij , ∀i ∈ [N], j ∈ [Ki] d c θ ∈ δij ∈ Dij , ∀i ∈ [N], j ∈ [Ki] (14) subject to d θe ij = δij , ∀i ∈ [N], j ∈ [Ki]  d  ui ∈ [0, i], ∀i ∈ [M]  n c o gei θe ; A − i + ui = 0, ∀i ∈ [M]. z, θ ,

Compared to problem (5), the introduction of auxiliary vari ables {ui} enables ADMM to incorporate black-box equality constraints as well as elementary white-box constraints. Sim ilar to Algorithm 1, the ADMM solution to problem (14) can Algorithm 2 Operator splitting from ADMM to solve problem

(14)

(with black-box constraints)

2 (t) n c(t+1) d(t+1) (t+1)o ρ d c d ρ XM  µi 2 θ , θe , u = arg min fe+ θe − b + IC(θ ) + IDe(θe ) + IU (u) + gei + ui − i + , 2 2 2 ρ θc,θed ,u i=1 (t+1) ρ 2 δ = arg min kδ − ak + ID(δ), 2 δ 2 (t+1) ρ XM  (t+1) 1 (t) 2 z = arg min fe+ IZ (z) + gei − i + ui + µi , z 2 ρ i=1

where the arguments of f˜ and g˜i are omitted for brevity, a and b have been defined in Algorithm 1, U = {u: u = {ui}, and µi is the Lagrangian multiplier corresponding to the equality constraint g˜i − i + ui = 0 in (14) and updated as d (t+1) (t) (t+1) c(t+1) µi = µi + ρ(gei(z , {θ , θe (t+1)}; A) − i + ui (t+1)) for ∀i ∈ [M].

be achieved by solving three sub-problems of similar nature, summarized in Algorithm 2 and derived in Appendix 2.

We remark that the integration of ADMM and gradient free operations was also considered in (Liu et al. 2018) and (Ariafar et al. 2017; 2019), where the former used random ized gradient estimator when optimizing a black-box smooth objective function, and the latter used Bayesian optimization (BO) as the internal solver to solve black-box optimization problems with black-box constraints. However, the afore mentioned works cannot directly be applied to tackling our considered AutoML problem, which requires a more involved splitting over hypeparameters and model selection variables. Moreover, different from (Ariafar et al. 2019), we handle the black-box inequality constraint through the reformulated equality constraint (13). By contrast, the work (Ariafar et al. 2019) introduced an indicator function for a black-box constraint and further handled it by modelling as a Bernoulli random variable.

## 4.3 Implementation and convergence

We highlight that our ADMM based scheme is not a single AutoML algorithm but rather a framework that can be used to mix and match different existing black-box solvers. This is es pecially useful since this enables the end-user to plug-in effi cient solvers tailored for the sub-problems (HPO & algorithm selection in our case). In addition to the above, the ADMM decomposition allows us to solve simpler sub-problems with a smaller number of optimization variables (a significantly re duced search space since (θ-min) only requires optimization over the active set of continuous variables). Unless specified otherwise, we adopt Bayesian optimization (BO) to solve the HPO (θ-min), e.g., (10). We use customized Thompson sam pling to solve the combinatorial multi-armed bandit problem, namely, the (z-min) for algorithm selection. We refer readers to Appendix 3 and 4 for more derivation and implementation details. In Appendix 5, we demonstrate the generalizability of ADMM to different solvers for (θ-min) and (z-min).The theoretical convergence guarantees of ADMM have been established under certain assumptions, e.g., convexity or smoothness (Boyd and others 2011; Hong and Luo 2017; Liu et al. 2018). Unfortunately, the AutoML problem vi olates these restricted assumptions. Even for non-ADMM based AutoML pipeline search, there is no theoretical conver gence established in the existing baselines to the best of our

knowledge. Empirically, we will demonstrate the improved convergence of the proposed scheme against baselines in the following section.

## 5 Empirical Evaluations

In this evaluation of our proposed framework, we demon strate three important characteristics: (i) the empirical per formance against existing AutoML toolkits, highlighting the empirical competitiveness of the theoretical formalism, (ii) the systematic capability to handle black-box constraints, enabling AutoML to address real-world ML tasks, and (iii) the flexibility to incorporate various learning procedures and solvers for the sub-problems, highlighting that our proposed scheme is not a single algorithm but a complete framework for AutoML pipeline configuration.

Data and black-box objective function. We consider 30 bi nary classification3datasets from the UCI ML (Asuncion and Newman 2007) & OpenML repositories (Bischl and others 2017), and Kaggle. We consider a subset of OpenML100 limited to binary classification and small enough to allow for meaningful amount of optimization for all baselines in the allotted 1 hour to ensure that we are evaluating the optimizers and not the initialization heuristics. Dataset details are in Ap pendix 6. We consider (1 − AUROC) (area under the ROC curve) as the black-box objective and evaluate it on a 80-20% train-validation split for all baselines. We consider AUROC since it is a meaningful predictive performance metric re gardless of the class imbalance (as opposed to classification error).

Comparing ADMM to AutoML baselines. Here we eval uate the proposed ADMM framework against widely used AutoML systems Auto-sklearn (Feurer et al. 2015) and TPOT (Olson and Moore 2016). This comparison is limited to black box optimization with analytic constraints only given by (1) since existing AutoML toolkits cannot handle black-box con straints explicitly. We consider SMAC based vanilla Auto sklearn ASKL4 (disabling ensembles and meta-learning), random search RND, and TPOT with a population of 50

3Our scheme applies to multiclass classification & regression.

4Meta-learning and ensembling in ASKL are preprocessing and postprocessing steps respectively to the actual black-box optimiza tion and can be applied to any optimizer. We demonstrate this for ADMM in Appendix 8. So we skip these aspects of ASKL here.

(instead of the default 100) to ensure that TPOT is able to process multiple generations of the genetic algorithm in the allotted time on all data sets. For ADMM, we utilize BO for (θ-min) and CMAB for (z-min) – ADMM(BO,Ba)5 .

Figure 1: Average rank (across 30 datasets) of mean performance across 10 trials – lower rank is better.

For all optimizers, we use scikit-learn algorithms (Pedregosa, Varoquaux, and others 2011). The functional modules and the algorithms (with their hyper-parameters) are presented in Table A3 in Appendix 7. We maintain par ity6 across the various AutoML baselines by searching over the same set of algorithms (see Appendix 7). For each scheme, the algorithm hyper-parameter ranges are set us ing Auto-sklearn as the reference7 . We optimize for 1 hour & generate time vs. incumbent black-box objective curves aggregated over 10 trials. Details on the complete setup are in Appendix 10. The optimization convergence for all 30 datasets are in Appendix 11. At completion, ASKL achieves the lowest mean objective (across trials) in 6/30 datasets, TPOT50 in 8/30, RND in 3/30 and ADMM(BO,Ba) in 15/30, showcasing ADMM’s effectiveness.

Figure 1 presents the overall performance of the different AutoML schemes versus optimization time. Here we consider the relative rank of each scheme (with respect to the mean objective over 10 trials) for every timestamp, and average this rank across 30 data sets similar to the comparison in

5 ρ on In this setup, ADMM has 2 parameters: (i) the penalty the augmented term, (ii) the loss upper-bound fˆ in the CMAB algorithm (Appendix 4). We evaluate the sensitivity of ADMM on these parameters in Appendix 9. The results indicate that ADMM is fairly robust to these parameters, and hence set ρ = 1 and fˆ = 0.7 (0) throughout. We start the ADMM optimization with λ = 0.

6ASKL and ADMM search over the same search space of fixed pipeline shape & order. TPOT also searches over different pipeline shapes & orderings because of the nature of its genetic algorithm. 7

github.com/automl/auto-sklearn/tree/master/autosklearn/pipeline/components

Figure 2: Best objective achieved by any constraint satisfying pipeline from running the optimization for 1 hour seconds with varying thresholds for the two constraints – lower is better (please view in color). Note the log-scale on the vertical axis.

Feurer et al. (2015). With enough time, all schemes outper form random search RND. TPOT50 performs worst in the beginning because of the initial start-up time involved in the genetic algorithm. ASKL and ADMM(BO,Ba) have compa rable performance initially. As the optimization continues, ADMM(BO,Ba) significantly outperforms all other baselines. We present the pairwise performance of ADMM with ASKL (figure 1b) & TPOT50 (figure 1c).

Rank

AutoML with black-box constraints. To demonstrate the capability of the ADMM framework to incorporate real-world black-box constraints, we consider the recent Home Credit Default Risk Kaggle challenge8 with the black-box objec tive of (1 − AUROC), and 2 black-box constraints: (i) (de ployment) Prediction latency tp enforcing real-time predic tions, (ii) (fairness) Maximum pairwise disparate impact dI (Calders and Verwer 2010) across all loan applicant age groups enforcing fairness across groups (see Appendix 12).

We run a set of experiments for each of the constraints: (i) fixing dI = 0.7, we optimize for each of the thresholds tp = {1, 5, 10, 15, 20} (in µs), and (ii) fixing tp = 10µs and we optimize for each of dI = {0.05, 0.075, 0.1, 0.125, 0.15}. Note that the constraints get less restrictive as the thresholds increase. We apply ADMM to the unconstrained problem (UCST) and post-hoc filter constraint satisfying pipelines to demonstrate that these constraints are not trivially satisfied. Then we execute ADMM with these constraints (CST). Using BO for (θ-min) & CMAB for (z-min), we get two variants – UCST(BO,Ba) & CST(BO,Ba). This results in (5 + 5)×2 = 20 ADMM executions, each repeated 10×.

Figure 2 presents the objective achieved by the optimizer when limited only to constraint satisfying pipelines. Figure 2a presents the effect of relaxing the constraint on tp while Figure 2b presents the same for the constraint on dI . As expected, the objective improves as the constraints relax. In both cases, CST outperforms UCST, with UCST approaching CST as the constraints relax. Figure 3 presents the constraint satisfying capability of the optimizer by considering the frac tion of constraint-satisfying pipelines found (Figure 3a & 3b for varying tp & dI respectively). CST again significantly out performs UCST, indicating that the constraints are non-trivial to satisfy, and that ADMM is able to effectively incorporate the constraints for improved performance.

8 www.kaggle.com/c/home-credit-default-risk

Flexibility & benefits from ADMM operator splitting. It is common in ADMM to solve the sub-problems to higher approximation in the initial iterations and to an increasingly lower approximation as ADMM progresses (instead of the same approximation throughout) (Boyd and others 2011). We demonstrate (empirically) that this adaptive ADMM produces expected gains in the AutoML problem. Moreover, we show the empirical gains of ADMM from (i) splitting the AutoML problem (1) into smaller sub-problems which are solved in an alternating fashion, & (ii) using different solvers for the differently structured (θ-min) and (z-min).

First we use BO for both (θ-min) and (z-min). For ADMM with a fixed approximation level (fixed ADMM), we solve the sub-problems with BO to a fixed number I = 16, 32, 64, 128 iterations, denoted by ADMMI(BO,BO) (e.g., ADMM16(BO,BO)). For adaptive ADMM, we start with 16 BO iterations for the sub-problems and progres sively increase it with an additive factor F = 8 & 16 with every ADMM iteration until 128 denoted by AdADMM F8(BO,BO) & AdADMM-F16(BO,BO) respectively. We op timize for 1 hour and aggregate over 10 trials.

Figure 4 presents optimization convergence for 1 dataset (fri-c2). We see the expected behavior – fixed ADMM with small I dominate for small time scales but saturate soon; large I require significant start-up time but dominate for larger time scales. Adaptive ADMM (F = 8 & 16) is able to match the performance of the best fixed ADMM at every time scale.Please refer to Appendix 13 for additional results.

Next, we illustrate the advantage of ADMM on operator splitting. We consider 2 variants, AdADMM-F16(BO,BO) and AdADMM-F16(BO,Ba), where the latter uses CMAB for (z-min). For comparison, we solve the complete joint problem (1) with BO, leading to a Gaussian Process with a large number of variables, denoted as JOPT(BO).

Figure 5 shows the optimization convergence for 1 dataset (fri-c2). The results indicate that the operator splitting in ADMM provides significant improvements over JOPT(BO), with ADMM reaching the final objective achieved by JOPT with significant speedup, and then further improving upon that final objective significantly. These improvements of ADMM over JPOT on 8 datasets are summarized in Table 1, indicating significant speedup (over 10× in most cases) and further improvement (over 10% in many cases).

Let us use SBa and SBO to represent the temporal speedup achieved by AdADMM(BO,Ba) and AdADMM(BO,BO)

Figure 3: Fraction of pipelines found satisfying constraints with optimization for 1 hour with varying thresholds for the 2 constraints – higher is better. Note the log-scale on the vertical axis.

Figure 4: Optimization time (in seconds) vs. median validation performance with the inter-quartile range over 10 trials on fri-c2 dataset – lower is better (please view in color). Note the log scale on both axes. See Appendix 13 for additional results.

Figure 5: Optimization time vs. median validation performance with the inter-quartile range over 10 trials on fri-c2 dataset – lower is better (please view in color). Note the log scale on both axes. See Appendix 14 for additional results.

(eliding “-F16”) respectively to reach the best objective of JOPT, and similarly use IBa and IBO to represent the objec tive improvement at the final converged point. Table 1 shows that between AdADMM(BO,BO) and AdADMM(BO,Ba), the latter provides significantly higher speedups, but the for mer provides higher additional improvement in the final ob jective. This demonstrates ADMM’s flexibility, for example, allowing choice between faster or more improved solution.

## 6 Conclusions

Posing the problem of joint algorithm selection and HPO for automatic pipeline configuration in AutoML as a formal mixed continuous-integer nonlinear program, we leverage the ADMM optimization framework to decompose this prob lem into 2 easier sub-problems: (i) black-box optimization with a small set of continuous variables, and (ii) a combinato rial optimization problem involving only Boolean variables. These sub-problems can be effectively addressed by existing AutoML techniques, allowing ADMM to solve the overall problem effectively. This scheme also seamlessly incorpo rates black-box constraints alongside the black-box objective. We empirically demonstrate the flexibility of the proposed ADMM framework to leverage existing AutoML techniques and its effectiveness against open-source baselines.

| Dataset |SBa |SBO |IBa |IBO |
| --- | --- | --- | --- | --- |
| Bank8FM |10× |2× |0% |5% |
| CPU small |4× |5× |0% |5% |
| fri-c2 |153× |25× |56% |64% |
| PC4 |42× |5× |8% |13% |
| Pollen |25× |7× |4% |3% |
| Puma8NH |11× |4× |1% |1% |
| Sylvine |9× |2× |9% |26% |
| Wind |40× |5× |0% |5% |

Table 1: Comparing ADMM schemes to JOPT(BO), we list the speedup SBa & SBO achieved by AdADMM(BO,Ba) & AdADMM(BO,BO) respectively to reach the best objective of JOPT, and the final objective improvement IBa & IBO (respectively) over the JOPT objective. These numbers are generated using the aggre gate performance of JOPT and AdADMM over 10 trials.

## References

Agrawal, S., and Goyal, N. 2012. Analysis of thompson sampling for the multi-armed bandit problem. In Conference on Learning Theory, 39–1.

Ariafar, S.; Coll-Font, J.; Brooks, D.; and Dy, J. 2017. An admm framework for constrained bayesian optimization. NIPS Workshop on Bayesian Optimization.

Ariafar, S.; Coll-Font, J.; Brooks, D.; and Dy, J. 2019. Ad mmbo: Bayesian optimization with unknown constraints us ing admm. Journal of Machine Learning Research 20(123):1– 26.

Asuncion, A., and Newman, D. 2007. UCI ML Repository. Bergstra, J., and Bengio, Y. 2012. Random search for hyper parameter optimization. JMLR 13(Feb):281–305.

Bergstra, J. S.; Bardenet, R.; Bengio, Y.; and Kégl, B. 2011. Algorithms for hyper-parameter optimization. In NeurIPS.

Bischl, B., et al. 2017. OpenML benchmarking suites and the OpenML100. arXiv:1708.03731.

Boyd, S., et al. 2011. Distributed optimization and statistical learning via the alternating direction method of multipliers. Foundations and Trends R in Machine Learning 3(1):1–122.

Calders, T., and Verwer, S. 2010. Three naive bayes ap proaches for discrimination-free classification. Data Mining and Knowledge Discovery 21(2):277–292.

Caruana, R.; Niculescu-Mizil, A.; Crew, G.; and Ksikes, A. 2004. Ensemble selection from libraries of models. In ICML.

Chen, B.; Wu, H.; et al. 2018. Autostacker: A compositional evolutionary learning system. In Proceedings of the Genetic and Evolutionary Computation Conference, 402–409. ACM.

Conn, A. R.; Scheinberg, K.; and Vicente, L. N. 2009. Intro duction to derivative-free optimization. SIAM.

Costa, A., and Nannicini, G. 2018. Rbfopt: an open-source library for black-box optimization with costly function evalu ations. Mathematical Programming Computation 10(4).

Drori, I.; Krishnamurthy, Y.; et al. 2018. Alphad3m: Machine learning pipeline synthesis. In AutoML Workshop at ICML.

Durand, A., and Gagné, C. 2014. Thompson sampling for combinatorial bandits and its application to online feature selection. In AAAI Workshops.

Falkner, S.; Klein, A.; and Hutter, F. 2018. BOHB: Robust and efficient hyperparameter optimization at scale. In ICML.

Feurer, M.; Klein, A.; Eggensperger, K.; Springenberg, J.; Blum, M.; and Hutter, F. 2015. Efficient and robust automated machine learning. In NeurIPS.

Friedler, S. A., et al. 2019. A comparative study of fairness enhancing interventions in machine learning. In Proceedings of the Conference on Fairness, Accountability, and Trans parency, 329–338. ACM.

Hong, M., and Luo, Z.-Q. 2017. On the linear convergence of the alternating direction method of multipliers. Mathematical Programming 162(1):165–199.

Hutter, F.; Hoos, H. H.; and Leyton-Brown, K. 2011. Se quential Model-based Optimization for General Algorithm Configuration. In International Conference on Learning and Intelligent Optimization. Springer-Verlag.

Jamieson, K., and Talwalkar, A. 2016. Non-stochastic best arm identification and hyperparameter optimization. In AIS TATS.

Komer, B.; Bergstra, J.; and Eliasmith, C. 2014. Hyperopt sklearn: automatic hyperparameter configuration for scikit learn. In ICML workshop on AutoML.

Kotthoff, L., et al. 2017. Auto-weka 2.0: Automatic model selection and hyperparameter optimization in weka. JMLR.

Larson, J.; Menickelly, M.; and Wild, S. M. 2019. Derivative free optimization methods. Acta Numerica 28:287–404.

Li, L.; Jamieson, K.; DeSalvo, G.; Rostamizadeh, A.; and Tal walkar, A. 2018. Hyperband: A novel bandit-based approach to hyperparameter optimization. JMLR 18(185):1–52.

Liu, S.; Kailkhura, B.; Chen, P.-Y.; Ting, P.; Chang, S.; and Amini, L. 2018. Zeroth-order stochastic variance reduction for nonconvex optimization. In NeurIPS.

Mohr, F.; Wever, M.; and Hüllermeier, E. 2018. ML-Plan: Automated machine learning via hierarchical planning. Ma chine Learning 107(8-10):1495–1515.

Olson, R. S., and Moore, J. H. 2016. TPOT: A tree-based pipeline optimization tool for automating machine learning. In Workshop on AutoML.

Pedregosa, F.; Varoquaux, G.; et al. 2011. Scikit-learn: Machine learning in Python. JMLR.

Rakotoarison, H.; Schoenauer, M.; and Sebag, M. 2019. Automated Machine Learning with Monte-Carlo Tree Search. In IJCAI.

Sabharwal, A., et al. 2016. Selecting near-optimal learners via incremental data allocation. In AAAI.

Shahriari, B.; Swersky, K.; Wang, Z.; Adams, R. P.; and De Freitas, N. 2016. Taking the human out of the loop: A review of bayesian optimization. Proceedings of the IEEE.

Snoek, J.; Larochelle, H.; and Adams, R. P. 2012. Practical bayesian optimization of machine learning algorithms. In NeurIPS.

Thornton, C.; Hoos, H. H.; Hutter, F.; and Leyton-Brown, K. 2012. Auto-weka: Automated selection and hyper-parameter optimization of classification algorithms. arXiv:1208.3719.

Vanschoren, J. 2018. Meta-learning: A survey. arXiv:1810.03548.

Williams, C. K., and Rasmussen, C. E. 2006. Gaussian processes for machine learning. MIT Press Cambridge, MA.

Yang, C.; Akimoto, Y.; Kim, D. W.; and Udell, M. 2019. OBOE: Collaborative filtering for AutoML model selection. In KDD.

Zhu, C.; Byrd, R. H.; Lu, P.; and Nocedal, J. 1997. Algorithm 778: L-bfgs-b: Fortran subroutines for large-scale bound constrained optimization. ACM TOMS 23(4):550–560.

## Appendices of ‘An ADMM Based Framework for AutoML Pipeline Confifigurations’ 1 Derivation of ADMM sub-problems in Table 1

ADMM decomposes the optimization variables into two blocks and alternatively minimizes the augmented Lagrangian function (7) in the following manner at any ADMM iteration t

n

d d c(t+1) (t+1)o  (t) c (t) (t)  θ , θθ = aarg mmin L z , θ , θθ , δ ,λ (A15) d θ c,θθ

n

d (t+1) (t+1)o  c(t+1) (t+1) (t)  δ , z = aarg mmin L z, θ , θθ , δ,λ (A16) δ,z

d (t+1) (t)  (t+1) (t+1) λ = λ + ρ θθ − δ . (A17)

Problem (A15) can be simplifified by removing constant terms to get

d d d n c(t+1) (t+1)o  z (t) , n θ c , θθ o ; A  + IC(θ c ) + + IDD(θθ ) (A18) θ , θθ = aarg mmin ff d θ c,θθ

2 d (t)>  (t)  ρ d (t) + λ θθ − δ + θθ − δ , 2 2 d d ρ d 2  (t) n c o  c = aarg mmin ff z , θ , θθ ; A + IC(θ ) + + IDD(θθ ) + + θθ − b (A19) d 2 2 θ c,θθ 1 (t) (t) where b = δ − λ . ρ

A similar treatment to problem (A16) gives us

n

d (t+1) (t+1)o  n c(t+1) (t+1)o  δ , z = aarg mmin ff z, θ , θθ ; A + IZ (z) (A20) δ,z 2 d d (t)>  (t+1)  ρ (t+1) + ID(δ) + + λ θθ − δ + θθ − δ , 2 2 d  n c(t+1) (t+1)o  = aarg mmin ff z, θ , θθ ; A + IZ (z) (A21) δ,z ρ d 1 2 (t+1) (t) + ID(δ) + + ka − δk where a = + λ .

2 ρ 2 θθ (A22)

This simplifification exposes the independence between z and δ, allowing us to solve problem (A16) independently for z and δ as:

δ

(t+1)

= aarg mmin

δ

ID(δ) + +

ρ

2

ka

−

δk

2

2

where

a =

θθ

d

(t+1)

+

1

ρ

λ

(t)

,

(A23)

d (t+1)  n c(t+1) (t+1)o  z = aarg mmin ff z, θ , θθ ; A + IZ (z). (A24) z

So we are able to decompose problem (3) into problems (A19), (A23) and (A24) which can be solved iteratively along with the (t) λ updates (see Table 1). 

2

## Derivation of ADMM sub-problems in Table 2

Defifining U = {u: u = {ui ∈ [0, i ]∀i ∈ [M]}}, we can go through the mechanics of ADMM to get the augmented Lagrangian with λ and µi∀i ∈ [M] as the Lagrangian multipliers and ρ > 0 as the penalty parameter as follows:

d d d  c   n c o  c L z, θ , θθ , δ, u,λ, µ = ff z, θ , θθ ; A + IZ (z) + + IC(θ ) + + IDD(θθ ) + + ID(δ) 2 d >   ρ d + λ θθ − δ + θθ − δ 2 2 M d IU (u) ++X µi  gegi  z, n θ c , θθ o ; A  − i + ui  (A25) i=1 M ρ + X   z, n c d o  2 θ 2 gegi , θθ ; A − i + ui . i=1

ADMM decomposes the optimization variables into two blocks for alternate minimization of the augmented Lagrangian in the following manner at any ADMM iteration t

n

θ

d d c(t+1) (t+1) (t+1)o  (t) c (t) (t) (t)  , θθ θ c,θθ , u = aarg mmin L z , θ , θθ , δ , u,λ , µ (A26) d ,u

n

d (t+1) (t+1)o  c(t+1) (t+1) (t+1) (t) (t)  δ , z = aarg mmin L z, θ , θθ , δ, u ,λ , µ (A27) δ,z

d (t+1) (t)  (t+1) (t+1) λ = λ + ρ θθ − δ (A28)

∀i

d (t+1) (t)  (t+1) c(t+1) (t+1) ∈ [M], µi = µi + ρ gegi(z , {θ , θθ (t+1)}; A) − i + ui . (A29)

Note that, unlike the unconstrained case, the update of the augmented Lagrangian multiplier µi requires the evaluation of the black-box function for the constraint gi . Simplifying problem (A26) gives us

d  (t) n c o  mind ff z , θ , θθ ; A d θ c,θθ ,u 2 # M 2 1 ρ " d   n c d o  (t) (t) + θθ − b + X gegi z , θ , θθ ; A − i + ui + µi 2 2 ρ (A30) i=1 c  θ ∈ CCij∀i ∈ [N], j ∈ [Ki ], idj d  d (t) 1 (t) subject to θθ ∈ DDij∀i ∈ [N], j ∈ [Ki ], where b = δ − λ , ij ρ

 ui ∈ [0, i ],

(t) which can be further split into active and inactive set of continuous variables based on the z as in the solution of problem (A19) (the θ-min problem). The main difference from the unconstrained case in problem (A19) (the θ-min problem) to note here is that the black-box optimization with continuous variables now has M new variables ui (M is the total number of black-box constraints) which are active in every ADMM iteration. This problem (A30) can be solved in the same manner as problem (A19) (θ-min) using SMBO or TR-DFO techniques.

Simplifying and utilizing the independence of

z

and

δ,

we can split problem (A27) into the following problem for

δ

ρ d 1 2 (t+1) (t) miδn kδ − ak subject to δij ∈ DDij∀i ∈ [N], j ∈ [KKi ] where a = θθ + λ , 2 δ 2 ρ

(A31)

which remains the same as problem (A23) (the δ-min problem) in the unconstrained case, while the problem for z becomes d

min

z

ff(z,

{θ

c(t+1)

,

θθ

(t+1)};

A)

M ρ  d 1 2 c(t+1) (t+1) (t) (A32) + X gegi(z, {θ , θθ (t+1)}; A) − i + ui + µi 2 ρ i=1 Ki subject to zi ∈ {0{0, 1} , 1 >zi = 11, ∀i ∈ [N].

The problem for z is still a black-box integer programming problem, but now with an updated black-box function and can be handled with techniques proposed for the combinatorial problem (A24) in the absence of black-box constraints (the z-min problem).  3

## Bayesian Optimization for solving the (θ-min) problem on the active set

Problem (10) ((θ-min) on the active set) is a HPO problem. This can be solved with Bayesian optimization (BO) (Shahriari et al. 2016). BO has become a core component of various AutoML systems (Snoek, Larochelle, and Adams 2012). For any black-box objective function f(θ) defifined on continuous variables θ ∈ CC, BO assumes a statistical model, usually a Gaussian (t) process (GP), for f. Based on the observed function values y = [f[f(θ (0)), . . . , f(θ )]>, BO updates the GP and determines the (t+1) next query point θ by maximizing the expected improvement (EI) over the posterior GP model. Specififically the objective f(θ) is modeled as a GP with a prior distribution f(·) ∼ N N (µ(·), κ(·, ·)), where κ(·, ·) is a positive defifinite kernel. Given the observed function values y, the posterior prob2ability of a new function evaluation f(θ) at iteration t+ 1 1 is modeled as a Gaussian 2 distribution with mean µ(θ) and variance σ (θ) (Shahriari et al. 2016, Sec. III-A), where

µ(θθˆ) = = κ >[Γ + σ I] −1y and σ (θθˆ) = = κ(θθˆ, θθˆ) − κ >[Γ + σ I] −1κ, (A33) 2 2 2 n n

(i) t (i) t where κ is a vector of covariance terms between θ and {θ } i=0, and Γ denotes the covariance of {θ } i=0, namely, Γij = (i) (j) 2 κ(θ , θ ), and σ is a small positive number to model the variance of the observation noise. n

Remark 1 TTo determine the GP model (A33), we choose the kernel ffunction κ(·, ·) as the ARD Matérn 5/2 kernel (Snoek, Larochelle, and Adams 2012; Shahriari et al. 2016),

√ √ 5 0 2 2 κ(x, x ) = = τ exp(− 5r)(1 + + 5r + r ) (A34) 0 3

0 2 0 2 d ffor two vectors x, x , where r = Pd i=1(xi−x ) 2/τ , and {ττi} are kernel parameters. WWe determine the GP hypper-parameters i τ i i=0 d ψ = {{ττi} i=0, σ2 n} by minimizing the negative log marginal likelihood log p(y|ψ) (Shahriari et al. 2016),

2 > 2 minimize log ddet(Γ + σ I) + + y Γ + σ I −1 y. (A35) n n ψ

(t+1) With the posterior model (A33), the desired next query point θ maximizes the EI acquisition function

(t+1) + θ = aarg mmax EI(θ) := y − f(θ)  I(f(θ) ≤ y +) (A36) {θ∈C}

+ +  y − µ   y − µ  + = aarg mmax (y − µ)Φ + σφ , (A37) {θ∈C} σ σ

+ (i) where y = mmini∈[t] f(θ ), namely, the minimum observed value, I(f(θ) ≤ y +) = = 1 1 if f(θ) ≤ y +, and 0 otherwise 2 (indicating that the desired next query point θ should yield a smaller loss than the observed minimum loss), and µ & σ are defifined in (A33), Φ denotes the cumulative distribution function (CDF) of the standard normal distribution, and φ is its probability distribution function (PDF). This is true because substituting (A33) into (A36) allows us to simplify the EI acquisition function as follows:

f(θ)−µ  y − µ  f 0=  + σ + 0 EI(θ) = Ef 0 (y − f 0σ − µ)I f ≤ σ + +  y − µ    y − µ  + 0 0 = (y(y − µ)Φ − σEf 0 f I f ≤ σ σ y+−µ +  y − µ  σ + Z 0 = (y(y − µ)Φ − σ f 0φ(f )dfdf0 σ Z −∞ + +  y − µ   y − µ  + = (y(y − µ)Φ + σφ , σ σ

where the last equality holds since R xφ(x)dx = −φ(x) + + C for some constant C. Here we omitted the constant C since it does not afffect the solution to the EI maximization problem (A37). With the aid of (A37), EI can be maximized via projected gradient ascent. In practice, a customized bound-constrained L-BFGS-B solver (Zhu et al. 1997) is often adopted.

## 4 Combinatorial Multi-Armed Bandit (CMAB) for (z-min) (problem (A24))

Algorithm A1

Thompson Sampling for CMAB with probabilistic rewards

1:

Input:

Beta distribution priors

α0

and

δ0,

maximum iterations

L,

upper bound

fˆ

of loss

f.

2:

Set:

nj

(k)

and

rj

(k)

as the cumulative counts and rewards respectively of arm

j

pulls at bandit iteration

k.

3:

for

k

←

1, 2, . . . , L

do

4:

for

all arms

j

∈

[K]

do

5:

αj

(k)

←

α0

+

rj

(k),

δj

(k)

←

δ0

+

nj

(k)

−

rj

(k).

6:

Sample

ωj

∼

Beta(αj

(k),

δj

(k)).

7:

end for

8:

Determine the arm selection scheme

z(k)

by solving

N i maximize X (zi) >ω subject to zi ∈ {0, 1} Ki , 1 >zi = 1, i ∈ [N], (A38) z i=1

1 N > i where ω = [(ω ) >, . . . ,(ω ) >] is the vector of {ωj}, and ω is its subvector limited to module i. Apply strategy z(k) and observe continuous reward re

9:

  f(k + 1)   re fˆ = 1 − min max , 0 , 1 (A39)

where

f(k + 1)

is the loss value after applying

z(k).

10:

Observe binary reward

r

∼

Bernoulli(re).

11:

for

all arms

j

∈

[K]

do

12:

Update

nj

(k + 1)

←

nj

(k) +

zj

(k).

13:

Update

rj

(k + 1)

←

rj

(k) +

zj

(k)r.

14:

end for

15:

end for

As mentioned earlier, problem (A24) can be solved as an integer program, but has two issues: (i) QN Ki black-box function i=1 queries would be needed in each ADMM iteration, and (ii) integer programming is difficult with the equality constraints PKi zij = 1∀i ∈ [N]. j=1

We propose a customized combinatorial multi-armed bandit (CMAB) algorithm as a query-efficient alternative by interpreting problem (A23) through combinatorial bandits: We are considering bandits due to the stochasticity in the algorithm selection arising from the fact that we train the algorithm in a subset of pipelines and not the complete combinatorially large set of all pipelines – the basic idea is to project an optimistic upper bound on the accuracy of the full set of pipelines using Thompson sampling. We wish to select the optimal N algorithms (arms) from K = PN Ki algorithms based on bandit feedback i=1 (‘reward’) r inversely proportional to the loss f. CMAB problems can be efficiently solved with Thompson sampling (TS) (Durand and Gagné 2014). However, the conventional algorithm utilizes binary rewards, and hence is not directly applicable to our case of continuous rewards (with r ∝ 1 − f where the loss f ∈ [0, 1] denotes the black-box objective). We address this issue by using “probabilistic rewards” (Agrawal and Goyal 2012).

i We present the customized CMAB algorithm in Algorithm A1. The closed-form solution of problem (A38) is given by z = 1 j i i for j = arg maxj∈[Ki] ω , and z = 0 otherwise. Step 9 of Algorithm A1 normalizes the continuous loss f with respect to its j j

upper bound fˆ (assuming the lower bound is 0), and maps it to the continuous reward re within [0, 1]. Step 10 of Algorithm A1 converts a probabilistic reward to a binary reward. Lastly, steps 11-12 of Algorithm A1 update the priors of TS for combinatorial bandits (Durand and Gagné 2014). For our experiments, we set α0 = δ0 = 10. We study the effect of fˆ on the solution of the (z-min) problem in Appendix 5.

5

## ADMM with different solvers for the sub-problems

We wish to demonstrate that our ADMM based scheme is not a single AutoML algorithm but rather a framework that can be used to mix and match different existing (and future new) black-box solvers. First we demonstrate the ability to plug in different solvers for the continuous black-box optimization involved in problem (10) (θ-min on the active set). We consider a search space containing 39 scikit-learn (Pedregosa, Varoquaux, and others 2011) ML algorithms allowing for over 6000 algorithm combinations. The 4 different modules and the algorithms (along with their number and types of hyper-parameters) in each of those modules is listed in Table A2 in section 7 of the supplement. For the solvers, we consider random search (RND), an off-the-shelf Gaussian process based Bayesian optimization (Williams and Rasmussen 2006) using scikit-optimize (BO), our implementation of a Gaussian process based Bayesian optimization (BO*)(see section 3 in the supplement for details), and RBFOpt (Costa and Nannicini 2018). We use a randomized algorithm selection scheme (z-min) – from each functional module, we randomly select an algorithm from the set of choices, and return the best combination found. The penalty parameter ρ for the augmented Lagrangian term in ADMM is set 1.0 throughout this evaluation.

Figure A1: Average performance (across 10 runs) of different solvers for the ADMM sub-problem (A19) (Please view in color).

Figure A2: Performance inter-quartile range of different solvers for the ADMM sub-problem (A19) (Please view in color).

We present results for 5 of the datasets in the form of convergence plots showing the incumbent objective (the best objective value found till now) against the wall clock time. Here tmax = 2048, n = 128, R = 10. The results are presented in figures A1 & A2. The results indicate that the relative performance of the black-box solvers vary between data sets. However, our goal here is not to say which is best, but rather to demonstrate that our proposed ADMM based scheme is capable of utilizing any solver for the (θ-min) sub-problem to search over a large space pipeline configurations.

For the algorithm selection combinatorial problem (z-min), we compare random search to a Thompson sampling (Durand and Gagné 2014) based combinatorial multi-armed bandit (CMAB) algorithm. We developed a customized Thompson sampling scheme with probabilistic rewards. We detail this CMAB scheme in Appendix 4 (Algorithm A1) and believe that this might be of independent interest. Our proposed CMAB scheme has two parameters: (i) the beta distribution priors α0, δ0 (set to 10), and (ii) the loss upper bound fˆ (which we vary as 0.3, 0.5, 0.7).

Figure A3: Average performance (across 10 runs) of different solvers for the ADMM sub-problem (A24) (please view in color).

We again consider results in the form of convergence plots showing the incumbent objective (the best objective value found till now) against the number of pipeline combinations tried (number of “arms pulled”) in figures A3 & A4. The results indicate

Figure A4: Performance inter-quartile range of different solvers for the ADMM sub-problem (A24) (Please view in color).

for large number of pulls, all schemes perform the same. However, on 2/5 datasets, CMAB(0.7) (and other settings) outperforms random search for small number of pulls by a significant margin. Random search significantly outperforms CMAB on the Ionosphere dataset. The results indicate that no one method is best for all data sets, but ADMM is not tied to a single solver, and is able to leverage different solvers for the (z-min) step.

## 6 Details on the data

We consider data sets corresponding to the binary classification task from the UCI machine learning repository (Asuncion and Newman 2007), OpenML and Kaggle. The names, sizes and sources of the data sets are presented in Table A1. There are couple of points we would like to explicitly mention here:

* While we are focusing on binary classification, the proposed ADMM based scheme is applicable to any problem (such as multiclass & multi-label classification, regression) since it is a black-box optimization scheme and can operate on any problem specific objective.

* We consider a subset of OpenML100 limited to binary classification and small enough to allow for meaningful amount of optimization for all baselines in the allotted 1 hour to ensure that we are evaluating the optimizers and not the initialization heuristics.

The HCDR data set from Kaggle is a subset of the data presented in the recent Home Credit Default Risk competition (https://www.kaggle.com/c/home-credit-default-risk). We selected the subset of 10000 rows and 24 features using the following steps:

* We only considered the public training set since that is only set with labels available

* We kept all columns with keyword matches to the terms “HOME”, “CREDIT”, “DEFAULT”, “RISK”, “AGE”, “INCOME”, “DAYS”, “AMT”.

* In addition, we selected the top 10 columns most correlated to the labels column.

* For this set of features, we randomly selected 10000 rows with ≤ 4 missing values in each rows while maintaining the original class ratio in the dataset.

Table A1: Details of the data sets used for the empirical evaluations. The ‘Class ratios’ column corresponds to the ratio of the two classes in the data set, quantifying the class imbalance in the data.

| Data |# rows |# columns |Source |Class ratio |
| --- | --- | --- | --- | --- |
| Sonar |208 |61 |UCI |1 : 0.87 |
| Heart statlog |270 |14 |UCI |1 : 0.8 |
| Ionosphere |351 |35 |UCI |1 : 1.79 |
| Oil spill |937 |50 |OpenML |1 : 0.05 |
| fri-c2 |1000 |11 |OpenML |1 : 0.72 |
| PC3 |1563 |38 |OpenML |1 : 0.11 |
| PC4 |1458 |38 |OpenML |1 : 0.14 |
| Space-GA |3107 |7 |OpenML |1 : 0.98 |
| Pollen |3848 |6 |OpenML |1 : 1 |
| Ada-agnostic |4562 |48 |OpenML |1 : 0.33 |
| Sylvine |5124 |21 |OpenML |1 : 1 |
| Page-blocks |5473 |11 |OpenML |1 : 8.77 |
| Optdigits |5620 |64 |UCI |1 : 0.11 |
| Wind |6574 |15 |OpenML |1 : 1.14 |
| Delta-Ailerons |7129 |6 |OpenML |1 : 1.13 |
| Ringnorm |7400 |21 |OpenML |1 : 1.02 |
| Twonorm |7400 |21 |OpenML |1 : 1 |
| Bank8FM |8192 |9 |OpenML |1 : 1.48 |
| Puma8NH |8192 |9 |OpenML |1 : 1.01 |
| CPU small |8192 |13 |OpenML |1 : 0.43 |
| Delta-Elevators |9517 |7 |OpenML |1 : 0.99 |
| Japanese Vowels |9961 |13 |OpenML |1 : 0.19 |
| HCDR |10000 |24 |Kaggle |1 : 0.07 |
| Phishing websites |11055 |31 |UCI |1 : 1.26 |
| Mammography |11183 |7 |OpenML |1 : 0.02 |
| EEG-eye-state |14980 |15 |OpenML |1 : 0.81 |
| Elevators |16598 |19 |OpenML |1 : 2.24 |
| Cal housing |20640 |9 |OpenML |1 : 1.46 |
| MLSS 2017 CH#2 |39948 |12 |OpenML |1 : 0.2 |
| 2D planes |40768 |11 |OpenML |1 : 1 |
| Electricity |45312 |9 |OpenML |1 : 0.74 |

## 7 Search space: Algorithm choices and hyper-parameters

In this section, we list the different search spaces we consider for the different empirical evaluations in section 5 of the paper.

## Larger search space

For the empirical evaluation of black-box constraints (section 5 (ii)), ADMM flexibity (section 5 (iii)) and Appendix 5, we consider 5 functional modules – feature preprocessors, feature scalers, feature transformers, feature selectors, and finally estimators. The missing handling and the categorical handling is always applied if needed. For the rest of the modules, there are 8, 11, 7 and 11 algorithm choices respectively, allowing for 6776 possible pipeline combinations. We consider a total of 92 hyperparamters across all algorithms. The algorithm hyper-parameter ranges are set using Auto-sklearn as the reference ( see https://github.com/automl/auto-sklearn/tree/master/autosklearn/pipeline/components).

Table A2: Overview of the scikit-learn feature preprocessors, feature transformers, feature selectors and estimators used in our empirical evaluation. The preprocessing is always applied so there is no choice there. Barring that, we are searching over a total of 8×11×7×11 = 6776 possible pipeline compositions.

| Module |Algorithm |# parameters |
| --- | --- | --- |
|   |Imputer |1d |
| Preprocessors |OneHotEncoder |none |
|   |None∗ Normalizer |none |
|   |  |none |
|   |QuantileTransformer |2d† |
|   |MinMaxScaler |  |
| Scalers ×8 |StandardScaler |none |
|   |RobustScaler |none |
|   |Binarizer |2c† , 2d |
|   |KBinsDiscretizer |2d |
|   |None |none |
|   |SparseRandomProjection |1c, 1d |
|   |GaussianRandomProjection RBFSampler |1d 1c, 1d |
|   |Nystroem |2c, 3d |
| Transformer ×11 |TruncatedSVD KernelPCA |2d 2c, 4d |
|   |FastICA |5d |
|   |FactorAnalysis |3d |
|   |PCA |1c, 1d |
|   |PolynomialFeatures |3d |
| Selector ×7 |None |none |
|   |SelectPercentile |1d |
|   |SelectFpr |1c |
|   |SelectFdr |1c |
|   |SelectFwe |1c |
|   |VarianceThreshold |1c |
|   |GaussianNB |none |
|   |QuadraticDiscriminantAnalysis |1c |
|   |GradientBoostingClassifier |3c, 6d |
|   |KNeighborsClassifier RandomForestClassifier |3d |
| Estimator ×11 |ExtraTreesClassifier |1c, 5d 1c, 5d |
|   |AdaBoostClassifier |1c, 2d |
|   |DecisionTreeClassifier |3c, 3d |
|   |GaussianProcessClassifier |2d |
|   |LogisticRegression |2c, 3d |
|   |MLPClassifier |2c, 5d |

∗None means no algorithm is selected and corresponds to a empty set of hyper † parameters. ‘d’ and ‘c’ represents discrete and continuous variables, respectively.

## Smaller search space for comparing to AutoML baselines

We choose a relatively smaller search space in order to keep an efficient fair comparison across all baselines, auto-sklearn, TPOT and ADMM, with the same set of operators, including all imputation and rescaling. However, there is a technical issue – many of the operators in Auto-sklearn are custom preprocessors and estimators (kitchen sinks, extra trees classifier preprocessor, linear svc preprocessors, fastICA, KernelPCA, etc) or have some custom handling in there (see https://github.com/automl/auto-sklearn/ tree/master/autosklearn/pipeline/components). Inclusion of these operators makes it infeasible to have a fair comparison across all methods. Hence, we consider a reduced search space, detailed in Table A3. It represents 4 functional modules with a choice of 6×3×6 = 108 possible method combinations (contrast to Table A2). For each scheme, the algorithm hyper-parameter ranges are set using Auto-sklearn as the reference (see https://github.com/automl/auto-sklearn/tree/master/autosklearn/pipeline/components).

Table A3: Overview of the scikit-learn preprocessors, transformers, and estimators used in our empirical evaluation comparing ADMM, auto-sklearn, TPOT. We consider a choice of 6 × 3 × 6 = 108 possible method combinations (see text for further details).

| Module |Algorithm |# parameters |
| --- | --- | --- |
| Preprocessors |1d none |Imputer OneHotEncoder |
| None∗ Normalizer QuantileTransformer MinMaxScaler StandardScaler RobustScaler |none none 2d† none none 2c† , 2d 2d none |Scalers ×6 |
| None PCA PolynomialFeatures |1c, 1d 1c, 2d none |Transformer ×3 |
| GaussianNB QuadraticDiscriminantAnalysis GradientBoostingClassifier Estimator ×6 KNeighborsClassifier RandomForestClassifier ExtraTreesClassifier |1c 3c, 6d 3d 1c, 5d 1c, 5d |  |

∗None

means no algorithm is selected and corresponds to a empty set of hyper

parameters.

†

‘d’ and ‘c’ represents discrete and continuous variables, respectively.

Note on parity between baselines. With a fixed pipeline shape and order, ADMM & ASKL are optimizing over the same search space by making a single selection from each of the functional modules to generate a pipeline. In contrast, TPOT can use multiple methods from the same functional module within a single pipeline with stacking and chaining due to the nature of the splicing/crossover schemes in its underlying genetic algorithm. This gives TPOT access to a larger search space of more complex pipelines featuring longer as well as parallel compositions, rendering the comparison somewhat biased towards TPOT. Notwithstanding this caveat, we consider TPOT as a baseline since it is a competitive open source AutoML alternative to ASKL, and is representative of the genetic programming based schemes for AutoML. We provide some examples of the complex pipelines found by TPOT in Appendix 16.

## 8 Learning ensembles with ADMM

We use the greedy selection based ensemble learning scheme proposed in Caruana et al. (2004) and used in Auto-sklearn as a post-processing step (Feurer et al. 2015). We run ASKL and ADMM(BO, Ba) for tmax = 300 seconds and then utilize the following procedure to compare the ensemble learning capabilities of Auto-sklearn and our proposed ADMM based optimizer:

* We consider different ensemble sizes e1 = 1 < e2 = 2 < e3 = 4 . . . < emax = 32.

•

* We perform library pruning on the pipelines found during the optimization run for a maximum search time tmax by picking only the emax best models (best relative to their validation score found during the optimization phase).

* Starting with the pipeline with the best sˆ as the first member of the ensemble, for each ensemble size ej , we greedily add the 0 pipeline (with replacement) which results in the best performing bagged ensemble (best relative to the performance sˆ on the j validation set Sv after being trained on the training set St).

• Once the ensemble members (possibly with repetitions) are chosen for any ensemble size ej , the ensemble members are retrained on the whole training set (the training + validation set) and the bagged ensemble is then evaluated on the unseen 0 held-out test set Sh to get s . We follow this procedure since the ensemble learning uses the validation set and hence cannot j be used to generate a fair estimate of the generalization performance of the ensemble.

* Plot the (ej , s0 ) pairs. j

* 0 • The whole process is repeated R = 10 times for the same T and ej s to get error bars for s j

.

For ADMM(BO,Ba), we implement the Caruana et al. (2004) scheme ourselves. For ASKL:SMAC3, we use the post-processing ensemble-learning based on the example presented in their documentation at https://automl.github.io/auto-sklearn/master/ examples/example_sequential.html.

Figure A5: Ensemble size vs. median performance on the test set and the inter-quartile range (please view in color). The Aquamarine and Blue curves correspond to ADMM(BO,Ba) and ASKL respectively.

The inter-quartile range (over 10 trials) of the test performance of the post-processing ensemble learning for a subset of the data sets in Table A1 is presented in Figure A5. The results indicate that the ensemble learning with ADMM is able to improve the performance similar to the ensemble learning in Auto-sklearn. The overall performance is driven by the starting point (the test error of the best single pipeline, corresponding to an ensemble of size 1) – if ADMM and Auto-sklearn have test objective values that are close to each other (for example, in Page-blocks and Wind), their performance with increasing ensemble sizes are very similar as well.

## 9 Parameter sensitivity check for ADMM

We investigate how sensitive our proposed approach is to the ADMM parameter ρ and CMAB parameter fˆ. For each parameter combination of ρ ∈ {0.001, 0.01, 0.1, 1, 10} and fˆ ∈ {0.5, 0.6, 0.7, 0.8, 0.9}, in Figure A6 we present the validation error (averaged over 10 trials) by running our approach on the HCDR dataset (see Appendix 6).

Figure A6: Validation error of our proposed ADMM-based approach against ADMM parameter ρ and CMAB parameter fˆ

For this experiment, the results indicate that a large ρ yields a slightly better performance. However, in general, our approach is not very sensitive to the choice of ρ and fˆ – the range of the objectives achieved are in a very small range. Based on this observation, we set ρ = 1 and fˆ = 0.7 in all our empirical evaluations of ADMM(BO,Ba) unless otherwise specified.

## 10 Details on the baselines and evaluation scheme

Evaluation scheme. The optimization is run for some maximum runtime T where each proposed configuration is trained on a set St and evaluated on Sv and the obtained score sˆ is the objective that is being minimized by the optimizer. We ensure that all the optimizers use the same train-validation split. Once the search is over, the history of attempted configurations is used to generate a search time vs. holdout performance curve in the following manner for N timestamps:

* For each timestamp ti , i = 1, . . . , N, tN = T, we pick the best validation score sˆi obtained by any configuration found by time ti from the start of the optimization (the incumbent best objective).

* Then we plot the (ti , sˆi) pairs.

* The whole above process is repeated R times for the same T, N and tis to get inter-quartile ranges for the curves.

* For the presented results, T = 3600 seconds, N = 256 and R = 10.

Parity with baselines. First we ensure that the operations (such as model training) are done single-threaded (to the extent possi ble) to remove the effects of parallelism in the execution time. We set OPENBLAS_NUM_THREADS and OMP_NUM_THREADS to 1 before the evaluation of ADMM and the other baselines. ADMM can take advantage of the parallel model-training much like the other systems, but we want to demonstrate the optimization capability of the proposed scheme independent of the underlying parallelization in model training. Beyond this, there are some details we note here regarding comparison of methods based on their internal implementation:

* For any time ti , if no predictive performance score (the objective being minimized) is available, we give that method the worst objective of 1.0 for ranking (and plotting purposes). After the first score is available, all following time stamps report the best incumbent objective. So comparing the different baselines at the beginning of the optimization does not really give a good view of the relative optimization capabilities – it just illustrates the effect of different starting heuristics.

* For ADMM, the first pipeline tried is Naive Bayes, which is why ADMM always has some reasonable solution even at the earliest timestamp.

* The per configuration run time and memory limits in Auto-sklearn are removed to allow Auto-sklearn to have access to the same search space as the ADMM variants.

* The ensembling and meta-learning capabilities of Auto-sklearn are disabled. The ensembling capability of Auto-sklearn is discussed further in Appendix 8.

* For ASKL, the first pipeline tried appears to be a Random Forest with 100 trees, which takes a while to be run. For this reason, there is no score (or an objective of 1.0) for ASKL until its objective suddenly drops to a more competitive level since Random Forests are very competitive out of the box.

* For TPOT, the way the software is set up (to the best of our understanding and trials), scores are only available at the end of any generation of the genetic algorithm. Hence, as with ASKL, TPOT do not report any scores until the first generation is complete (which implies worst-case objective of 1.0), and after that, the objective drops significantly. For the time limit considered (T = 3600 seconds), the default population size of 100 set in TPOT is unable to complete a multiple generations on most of the datasets. So we reduce the population size to 50 to complete a reasonable number of generations within the set time.

* As we have discussed earlier, TPOT has an advantage over ASKL and ADMM – TPOT is allowed to use multiple estimators, transformers and preprocessors within a single pipeline via stacking and chaining due to the nature of the splicing and crossover schemes in its underlying genetic algorithm. This gives TPOT access to a larger search space of more complex pipelines featuring longer as well as parallel compositions; all the remaining baselines are allowed to only use a single estimator, transformers and preprocessor. Hence the comparison is somewhat biased towards TPOT, allowing TPOT to potentially find a better objective in our experimental set up. If TPOT is able to execute a significant number of generations, we have observed in many cases that TPOT is able to take advantage of this larger search space and produce the best performance.

* Barring the number of generations (which is guided by the maximum run time) and the population size (which is set to 50 to give us TPOT50), the remaining parameters of mutation rate, crossover rate, subsample fraction and number of parallel threads to the default values of 0.9, 0.1, 1.0 and 1 respectively.

Random search (RND) is implemented based on the Auto-sklearn example for random search at https://automl.github.io/auto sklearn/master/examples/example_random_search.html.

Compute machines. All evaluations were run single-threaded on a 8 core 8GB CentOS virtual machines.

## 11 Convergence plots for all data sets for all AutoML baselines.

g

Object

-

Figure A7: Search/optimization time vs. median validation performance with the inter-quartile range over 10 trials (please view in color). The curves colored Aquamarine, Grey, Blue and Black correspond respectively to ADMM(BO,Ba), RND, ASKL and TPOT50.

## 12 Computing the group-disparity fairness metric with respect to classification metric ε

Computing the black-box function. The black-box objective f(z, θ, A) is computed as follows for holdout-validation with some metric ε (the metric ε can be anything such as zero-one loss or area under the ROC curve):

* Let m be the pipeline specified by (z, θ)

* Split data set A into training set At and validation set Av

* Train the pipeline m with training set At to get mAt

* Evaluate the trained pipeline mAt on the validation set Av as follows:

ε (At, Av) = ε ({(y, mAt (x)) ∀(x, y) ∈ Av}), (A40)

where mAt (x) is the prediction of the trained pipeline mAt on any test point x with label y and

f(z, θ,

A)

= ε

(At, Av).

(A41)

For k-fold cross-validation, using the above notation, the objective is computed as follows:

• Split data set A into training set Ati and validation set Avi for each of the i = 1, . . . , k folds

* For a pipeline m specified with (z, θ), the objective is computed as

k 1 f(z, θ, A) = X ε (Ati , Avi ). (A42) k i=1

NOTE. The splitting of the data set A in training/validation pairs (At, Av) should be the same across all evaluations of (z, θ). Similarly, the k-fold splits should be the same across all evaluations of (z, θ).

Computing group disparate impact. Continuing with the notation defined in the previous subsection, for any given (test/validation) set Av, assume that we have a (probably user specified) “protected” feature d and a grouping Gd(Av) = {A1, A2, . . .} of the set Av based on this feature (generally, Aj ∩ Ak = ∅∀j =6 k and ∪Aj∈Gd(A)Aj = Av). Then, given the objective function f corresponding to the metric ε, the corresponding group disparate impact with holdout validation is given as

p(z, θ, A) = max ε (At, Aj ) − min ε (At, Aj ) (A43) Aj∈Gd(Av) Aj∈Gd(Av)

For k-fold cross-validated group disparate impact with the grouping per fold as Gd(Avi ) = {Ai,1, Ai,2, . . .}, we use the following:

k 1   p(z, θ, A) = X max ε (Ati , Ai,j ) − min ε (Ati , Ai,j ) (A44) k Ai,j∈Gd(Avi ) Ai,j∈Gd(Avi ) i=1

Example considered here:

* Dataset A: Home credit default risk Kaggle challenge

* Metric ε: Area under ROC curve

* Protected feature d: DAYS_BIRTH

* Grouping Gd based on d: Age groups 20-30, 30-40, 40-50, 50-60, 60-70

## 13 Benchmarking Adaptive ADMM

It is common in ADMM to solve the sub-problems to higher level of approximation in the initial ADMM iterations and to an increasingly smaller levels of approximation as the ADMM progresses (instead of the same level of approximation for all ADMM iterations). We make use of this same adaptive ADMM and demonstrate that, empirically, the adaptive scheme produces expected gains in the AutoML problem as well.

In this empirical evaluation, we use BO for solving both the (θ-min) and the (z-min) problems. For ADMM with a fixed level of approximation (subsequently noted as fixed ADMM), we solve the sub-problems to a fixed number I of BO iterations with I = 16, 32, 64, 128 (also 256 for the artificial objective described in Appendix 15)) denoted by ADMMI(BO,BO) (for example, ADMM16(BO,BO)). For ADMM with varying level of approximation, we start with 16 BO iterations for the sub-problems and progressively increase it with an additive factor F = 8 or 16 with every ADMM iteration until 128 (until 256 for the artificial objective) denoted by AdADMM-F8(BO,BO) and AdADMM-F16(BO,BO) respectively. The optimization is run for 3600 seconds for all the data sets and for 1024 seconds for the artificial objective function. The convergence plots are presented in Figure A8.

ADMM16(BO,

BO)

ADMM256(BO.

BO)

ADMM32(BO,

BO)

AdADMM-F16(BO,

BO)

ADMM64(BO,

BO)

AdADMM-F8(BO,

BO)

-

ADMM128(BO,

BO)

Obiective

Median

2°

2°

27

2-

2°

Optimization

time

seconds)

(in

Figure A8: Search/optimization time (in seconds) vs. median validation performance with the inter-quartile range over 10 trials (please view in color and note the log scale on both the horizontal and vertical axes).

The figures indicate the expected behavior – fixed ADMM with small I dominate for small optimization time scale but saturate soon while fixed ADMM with large I require a significant amount of startup time but then eventually lead to the best performance for the larger time scales. Adaptive ADMM (for both values of F) appears to somewhat match the performance of the best fixed ADMM for every time scale. This behavior is exemplified with the artificial black-box objective (described in Appendix 15) but is also present on the AutoML problem with real datasets.

## 14 Evaluating the benefits of problem splitting in ADMM

In this empirical evaluation, we wish to demonstrate the gains from (i) splitting the AutoML problem (1) into smaller sub problems which are solved in an alternating fashion, and (ii) using different solvers for the differently structured (θ-min) and (z-min) problems. First, we attempt to solve the complete joint optimization problem (1) with BO, leading to a Gaussian Process with a large number of variables. We denote this as JOPT(BO). Then we utilize adaptive ADMM where we use BO for each of the (θ-min) and (z-min) problems in each of the ADMM iteration, denoted as AdADMM-F16(BO,BO). Finally, we use adaptive ADMM where we use BO for each of the (θ-min) problem and Combinatorial Multi-Armed Bandits (CMAB) for the (z-min) problem, denoted as AdADMM-F16(BO,Ba). For the artificial black-box objective (described in Appendix 15), the optimization is run for 1024 seconds. For the AutoML problem with the actual data sets, the optimization is run for 3600 seconds. The convergence of the different optimizers are presented in Figure A9.

JOPT(BO)

AdADMM-F16(BO,

BO)

AdADMM-F16(BO,

Figure A9: Search/optimization time vs. median validation performance with the inter-quartile range over 10 trials (please view in color and note the log scale on both the horizontal and vertical axes).

The results for the artificial objective is a case where the black-box optimization dominates the optimization time (since the black-box evaluation is cheap). In this case, both versions of the adaptive ADMM significantly outperforms the single BO (JOPT(BO)) for the whole problem 2 seconds onwards, demonstrating the advantage of the problem splitting in ADMM. Between the two versions of the adaptive ADMM, AdADMM(BO,Ba) (Bandits for (z-min)) outperforms AdADMM(BO,BO) (BO for (z-min)). This is potentially because BO is designed for continuous variables and is mostly suited for the (θ-min) problem, whereas the Bandits interpretation is better suited for the (z-min) problem. By the end of the optimization time budget, AdADMM(BO,Ba) improves the objective by around 31% over JOPT(BO) (5% over AdADMM(BO,BO)), and achieves the objective reached by JOPT(BO) with a 108× speedup (AdADMM(BO,BO) with a 4× speedup).

On the AutoML problem with real data sets, the optimization time is mostly dominated by the black-box evaluation, but even in this case, the problem splitting with ADMM demonstrates significant gains over JOPT(BO). For example, on the fri-c2 dataset,

the results indicate that the operator splitting in ADMM allows it to reach the final objective achieved by JOPT with over 150× speedup, and then further improves upon that final objective by over 50%. On the Pollen dataset, we observe a speedup of around 25× with a further improvement of 4%. Table A4 & A5 summarize the significant gains from the problem splitting in ADMM.

| Dataset |Speedup |Improvement |
| --- | --- | --- |
| Artificial |108× |31% |
| Bank8FM |10× |0% |
| CPU small |4× |0% |
| fri-c2 |153× |56% |
| PC4 |42× |8% |
| Pollen |25× |4% |
| Puma8NH |11× |1% |
| Sylvine |9× |9% |
| Wind |40× |0% |

Table A4: Comparing AdADMM(BO,Ba) to JOPT(BO), we list the speedup achieved by AdADMM(BO,Ba) to reach the best objective of JOPT(BO), and any further improvement in the objective. These numbers are generated using the aggregate performance of JOPT and AdADMM over 10 trials.

| Dataset |Speedup |Improvement |
| --- | --- | --- |
| Artificial |39× |27% |
| Bank8FM |2× |5% |
| CPU small |5× |5% |
| fri-c2 |25× |64% |
| PC4 |5× |13% |
| Pollen |7× |3% |
| Puma8NH |4× |1% |
| Sylvine |2× |26% |
| Wind |5× |5% |

Table A5: Comparing AdADMM(BO,BO) to JOPT(BO), we list the speedup achieved by AdADMM(BO,BO) to reach the best objective of JOPT(BO), and any further improvement in the objective. These numbers are generated using the aggregate performance of JOPT and AdADMM over 10 trials.

## 15 Artificial black-box objective

We wanted to devise an artificial black-box objective to study the behaviour of the proposed scheme that matches the properties of the AutoML problem (1) where

1. The same pipeline (the same algorithm choices z and the same hyperparameters θ always gets the same value.

* 2. The objective is not convex and possibly non-continuous.

* 3. The objective captures the conditional dependence between zi and θij – the objective is only dependent on the hyper-parameters θij if the corresponding zij = 1.

* 4. Minor changes in the hyper-parameters θij can cause only small changes in the objective.

* 5. The output of module i is dependent on its input from module i − 1.

Novel artificial black-box objective. To this end, we propose the following novel black-box objective:

• For each (i, j), i ∈ [N], j ∈ [Ki ], we fix a weight vector wij (each entry is a sample from N (0, 1)) and a seed sij . 0.

* We set f0 =

* For each module i, we generate a value

vi

=

X

j

zij

w>

ijθij

1

T

θij

which only depends on the θij corresponding to the zij = 1, and the denominator ensures that the number (or range) of the hyper-parameters does not bias the objective towards (or away from) any particular algorithm.

* We generate n samples {fi,1, . . . , fi,n} ∼ N (fi−1, vi) with the fixed seed sij , ensuring that the same value will be produced for the same pipeline.

* fi = max |fi,m|. m=1,...,n

The basic idea behind this objective is that, for each operator, we create a random (but fixed) weight vector wij and take a weighted normalized sum of the hyper-parameters θij and use this sum as the scale to sample from a normal distribution (with a fixed seed sij ) and pick the maximum absolute of n (say 10) samples. For the first module in the pipeline, the mean of the distribution is f0 = 0.0. For the subsequent modules i in the pipeline, the mean fi−1 is the output of the previous module i − 1. This function possesses all the aforementioned properties of the AutoML problem (1).

In black-box optimization with this objective, the black-box evaluations are very cheap in contrast to the actual AutoML problem where the black-box evaluation requires a significant computational effort (and hence time). However, we utilize this artificial objective to evaluate ADMM (and other baselines) when the computational costs are just limited to the actual derivative-free optimization.

## 16 TPOT pipelines: Variable length, order and non-sequential

The genetic algorithm in TPOT does stitches pipelines together to get longer length as well as non-sequential pipelines, using the same module multiple times and in different ordering. Given the abilities to

* i. have variable length and variable ordering of modules,

* ii. reuse modules, and

* iii. have non-sequential parallel pipelines,

TPOT does have access to a much larger search space than Auto-sklearn and ADMM. Here are some examples for our experiments:

## Sequential, length 3 with 2 estimators

Input --> PolynomialFeatures --> KNeighborsClassifier --> GaussianNB

GaussianNB(

KNeighborsClassifier(

PolynomialFeatures(

input_matrix,

PolynomialFeatures__degree=2, PolynomialFeatures__include_bias=False, PolynomialFeatures__interaction_only=False ), KNeighborsClassifier__n_neighbors=7, KNeighborsClassifier__p=1, KNeighborsClassifier__weights=uniform ) ) Sequential, length 4 with 3 estimators Input --> PolynomialFeatures --> GaussianNB --> KNeighborsClassifier --> GaussianNB GaussianNB( KNeighborsClassifier( GaussianNB( PolynomialFeatures( input_matrix, PolynomialFeatures__degree=2, PolynomialFeatures__include_bias=False, PolynomialFeatures__interaction_only=False ) ), KNeighborsClassifier__n_neighbors=7, KNeighborsClassifier__p=1, KNeighborsClassifier__weights=uniform ) ) Sequential, length 5 with 4 estimators

Input

--> RandomForestClassifier

--> RandomForestClassifier

--> GaussianNB

--> RobustScaler

--> RandomForestClassifier

RandomForestClassifier(

RobustScaler(

GaussianNB(

RandomForestClassifier(

RandomForestClassifier(

input_matrix,

RandomForestClassifier__bootstrap=False,

RandomForestClassifier__criterion=gini,

RandomForestClassifier__max_features=0.68,

RandomForestClassifier__min_samples_leaf=16,

RandomForestClassifier__min_samples_split=13,

RandomForestClassifier__n_estimators=100

),

RandomForestClassifier__bootstrap=False,

RandomForestClassifier__criterion=entropy,

RandomForestClassifier__max_features=0.9500000000000001,

RandomForestClassifier__min_samples_leaf=2,

RandomForestClassifier__min_samples_split=18,

RandomForestClassifier__n_estimators=100

)

)

),

RandomForestClassifier__bootstrap=False,

RandomForestClassifier__criterion=entropy,

RandomForestClassifier__max_features=0.48,

RandomForestClassifier__min_samples_leaf=2,

RandomForestClassifier__min_samples_split=8,

RandomForestClassifier__n_estimators=100

) Non-sequential

Combine[ Input, Input --> GaussianNB --> PolynomialFeatures --> Normalizer ] --> RandomForestClassifier RandomForestClassifier( CombineDFs( input_matrix, Normalizer( PolynomialFeatures( GaussianNB( input_matrix ), PolynomialFeatures__degree=2, PolynomialFeatures__include_bias=True, PolynomialFeatures__interaction_only=False ), Normalizer__copy=True, Normalizer__norm=l2 ) ), RandomForestClassifier__bootstrap=False, RandomForestClassifier__criterion=entropy, RandomForestClassifier__max_features=0.14, RandomForestClassifier__min_samples_leaf=7, RandomForestClassifier__min_samples_split=8, RandomForestClassifier__n_estimators=100

)