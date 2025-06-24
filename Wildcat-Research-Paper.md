## arXiv:2205.12595v1 [cs.RO] 25 May 2022

## Wildcat: Online Continuous-Time

## 3D Lidar-Inertial SLAM

### Milad Ramezani,Member, IEEE,, Kasra Khosoussi, Gavin Catt,Member, IEEE,Peyman Moghadam,Senior

### Member, IEEE,Jason Williams,Senior Member, IEEE,Paulo Borges,Senior Member, IEEE,Fred

### Pauling,Member, IEEE, Navinda Kottege,Senior Member, IEEE

```
Abstract—We present Wildcat, a novel online 3D lidar-inertial
SLAM system with exceptional versatility and robustness. At its
core, Wildcat combines a robust real-time lidar-inertial odometry
module, utilising a continuous-time trajectory representation,
with an efficient pose-graph optimisation module that seam-
lessly supports both the single- and multi-agent settings.The
robustness of Wildcat was recently demonstrated in the DARPA
Subterranean Challenge where it outperformed other SLAM sys-
tems across various types of sensing-degraded and perceptually
challenging environments. In this paper, we extensively evaluate
Wildcat in a diverse set of new and publicly available real-world
datasets and showcase its superior robustness and versatility over
two existing state-of-the-art lidar-inertial SLAM systems.
Index Terms—3D Lidar-Inertial SLAM, Localisation and
Mapping, Collaborative SLAM
```
#### I. INTRODUCTION

# S

```
IMULTANEOUS Localisation and Mapping (SLAM) is
the backbone of robotics downstream tasks such as robot
navigation in unknown GPS-denied environments. Among
existing solutions to SLAM, lidar-inertial systems are highly
popular due to their robustness, precision, and high-fidelity
maps. Beyond robotics applications, these systems also hold
the promise of providing scalable and low-cost alterna-
tive to conventional mapping and surveying systems (used
ine.g., construction) with comparable precision. During the
last two decades and with the advent of affordable 3D lidars,
many 3D lidar-inertial SLAM systems have been proposed;
see [1–5] and references therein.
Despite tremendous progress in recent years, designing a
robust and versatile lidar-inertial SLAM system remains a
challenge. In particular, designing features that can be reliably
detected and matched in a wide range of environments is
a difficult task. Additionally, lidar SLAM systems must be
able to account for the effects of platform’s motion and the
mechanical actuation within the sensor on lidar points (i.e.,
motion distortion). Although this issue can be mitigated by
incorporating data from an Inertial Measurement Unit (IMU),
fusion of asynchronous data from lidar and IMU presents
additional technical challenges.
This paper presents Wildcat, a state-of-the-art online 3D
lidar-inertial SLAM system, and showcases its exceptional
versatility and robustness over prior state-of-the-art systems
```
(^1) The authors are with the Robotics and Autonomous Systems
Group, DATA61, CSIRO, Brisbane, QLD 4069, Australia. E-mails:
firstname.lastname@csiro.au
Fig. 1:Top: Wildcat can map large-scale environments, including
various types of structured and unstructured areas indoors and
outdoors.Bottom: Two close-up views from the challenging areas
where the state-of-the-art systems struggle to converge (see Sec.
VI-D).
through extensive experimental evaluation and carefully de-
signed case studies. At its core, Wildcat combines anon-
lineimplementation of concepts from the pioneering (albeit
offline) odometry system proposed in [2] with a pose-graph
optimisation module, efficiently allowing to map large-scale
environments as seen in Fig. 1. Thanks to its modular design,
Wildcat also seamlessly supports decentralised collaborative
multi-agent localisation and mapping where agents exchange
their submaps via peer-to-peer communication and indepen-
dently optimise the collective pose graph.
Wildcat has been heavily field tested (i) in various types of
environments such as buildings, urban roads, mines, caves,
farms and forests; and (ii) on various types of platforms
including handheld, ground vehicles (e.g., cars, legged robots,
and tracked vehicles), and aerial robots. Most recently, the
robustness and versatility of Wildcat were demonstrated inthe
DARPA Subterranean Challenge where it outperformed other
state-of-the-art SLAM systems in a wide range of perceptually


challenging and sensing-degraded subterranean environments
(e.g., due to dust and smoke). Specifically, it was reported
by DARPA that the map produced by Wildcat using a team
of four heterogeneous robots in the Final Event had “0%
deviation” and “91% coverage”,^1 where deviation is defined
as the percentage of points in the submitted point cloud that
are farther than one meter from the points in the surveyed
point cloud map.
The main contributions of this paper are the following:

- We present Wildcat, a highly robust and versatile state-of-
    the-art lidar-inertial SLAM system. This paper provides
    a detailed technical description of Wildcat beyond the
    broad non-technical overview previously presented in [6].
- We demonstrate the robustness and versatility of Wildcat
    through carefully designed experiments. This includes
    quantitatively comparisons against two other state-of-the-
    art lidar-inertial systems [3, 7] on a publicly available
    dataset [8] and two unique new large-scale multi-domain
    datasets with over 60 accurately surveyed landmarks.

Outline

The remainder of this paper is organised as follows. We
first introduce the necessary notation below. We then review
existing lidar-based SLAM systems in Section II. An overview
of Wildcat’s core components is presented in Section III. This
is followed by a detailed description of Wildcat’s odometry
and pose-graph optimisation modules in Sections IV and V,
respectively. The results of extensive experimental evaluations
and quantitative comparisons against the state of the art are
presented in Section VI. Finally, we conclude the paper in
Section VII where we discuss future research directions.

Notation

We use[n],{ 1 , 2 ,... , n}to refer to the set of natural
numbers up to n. Bold lower- and upper-case letters are
generally reserved for vectors and matrices, respectively. The
standard inner product onRnis written as〈·,·〉. The special
Euclidean and special orthogonal groups are denoted bySE(3)
andSO(3), respectively. We useso(3)to refer to the Lie
algebra associated toSO(3). Matrix exponential and logarithm
are denoted byexpandlog, respectively. The hat operator
(·)∧:R^3 →so(3)gives the natural representation of vectors
inR^3 as 3 × 3 skew-symmetric matrices. The inverse of hat
operator is denoted by(·)∨:so(3)→R^3. Finally, the linear-
interpolation operatorLinInterpolateis defined as follows,

```
LinInterpolate:Rn×Rn×[0,1]→Rn
(x,y, α)7→αx+ (1−α)y. (1)
```
#### II. RELATED WORK

One of the most popular and influential lidar-inertial-based
systems is Lidar Odometry and Mapping (LOAM) [9, 10].
Assuming constant angular and linear velocity during a sweep,
LOAM linearly interpolates the pose transform at a high
frequency (10 Hz) but with low fidelity over the course of

(^1) https://youtu.be/SyjeIGCHnrU?t=
a sweep. By minimising the distance between corresponding
edge point and planar point features extracted in one sweep
and the next sweep as evolving, ego-motion is estimated
iteratively until convergence. Later, at a lower rate (1 Hz),
features of the frontier sweep which are deskewed by the
odometry algorithm are matched with the map generated on
the fly to estimate the sweep pose in the map frame. In [11] the
authors propose a computationally efficient framework based
on LOAM that can run at 20 Hz. It deals with lidar distortion
in a non-iterative two-stage method by first computing the
distortion from frame-to-frame matching and then updating
it in the frame-to-map matching step once the current pose
is estimated in an iterative pose optimisation. In contrast
to the loosely coupled approach proposed in LOAM, LIO-
Mapping [12] utilises atightly-coupledmethod in which lidar
and inertial measurements are fused in a joint optimisation
problem.
LeGo-LOAM [13] and LIO-SAM [3] are two popular lidar-
inertial SLAM systems based on LOAM that use pose-graph
optimisation. Pose-graph optimisation enables these methods
to remove drift due to accumulated odometry error and cre-
ate globally consistent maps by incorporating loop closures.
In particular, LIO-SAM is a tightly-coupled keyframe-based
online system that combines lidar odometry with IMU preinte-
gration, loop closures, and (if available) GPS factors via pose-
graph optimisation. LT-Mapper [14] builds on LIO-SAM and
uses Scan Context (SC) [15] for loop closure detection. We
compare Wildcat with LIO-SAM in Sec. VI. Another tightly-
coupled lidar-inertial SLAM system is IN2LAMA [16], an
offline system that addresses the lidar distortion problem by
interpolating IMU measurements using Gaussian Process (GP)
regression.
Filtering-based frameworks such as LINS [17], FAST-
LIO [1], and its successor FAST-LIO2 [7] tightly couple IMU
and lidar measurements. In particular, FAST-LIO2 [7] is an
odometry and mapping system based on iterated Kalman filter
in which raw lidar points are efficiently registered to the map.
FAST-LIO2 uses the backward propagation step introduced
in [1] to mitigate lidar distortion. We compare our method
against FAST-LIO2 in Sec. VI.
Another line of work explores the idea of using complex
and expressive continuous-time (CT) representations of robot
trajectory to address the lidar distortion problem (by query-
ing robot pose at any desired time), and also to facilitate
fusion of asynchronous measurements obtained from sensors
at different rates such as IMU and lidar [2, 18–20]; see
[21, 5] and references therein for a discussion of various CT-
based frameworks. In particular, B-splines [22, 2, 23] and
approaches based on GP regression [24] are two popular
choices. For example, [25] applies a GP-based approach [24]
to obtain the robot poses at measurement times in a lidar-based
visual odometry system. Recently, Droeschel and Behnke [26]
propose a CT hierarchical method for 3D lidar SLAM that,
similar to our work, combines local mapping with pose-graph
optimisation. The authors in [26] use Spline Fusion [23] to
address the lidar distortion problem. More recently, Parket al.
[21] proposed a map-centric SLAM system (ElasticLiDAR++)
which uses a CT map deformation method to maintain a


globally consistent map without relying on global trajectory
optimisation. Our lidar-inertial odometry is an online imple-
mentation of the concepts introduced in the offline systems
proposed by Bosse and Zlot [22] and Bosseet al.[2] based
on cubic B-spline interpolation.

#### III. WILDCATOVERVIEW

```
Wildcat is composed of the following two main modules:
1) A sliding-window lidar-inertial odometry and local map-
ping module (see Section IV), hereafter referred to as
Wildcat odometry. Wildcat odometry is designed to effi-
ciently integrate asynchronous IMU and lidar measure-
ments using continuous-time representations of trajectory
and to mitigate the distortion in map due to the motion
of the sensor.
2) A modern pose-graph optimisation (PGO) module. By
leveraging the odometry solution and local maps (Wild-
catsubmaps) produced by Wildcat odometry, the robot
trajectory and environment map are optimised at a global
scale. Wildcat merges submaps with sufficient overlap to
reduce pose-graph nodes, effectively mapping large-scale
environments (see Section V).
```
Fig. 2 displays Wildcat’s pipeline when running on a single
agent. In the following sections, we describe each module in
detail.

#### IV. WILDCATODOMETRY

Wildcat odometry is a real-time implementation of a num-
ber of concepts from [2]. This module processes data in a
sliding-window fashion. Thekth time windowWkis a fixed-
length time interval obtained by sliding the previous time
windowWk− 1 forward by a fixed amount. We now describe
the key steps taken by Wildcat odometry during thekth time
windowWk.

A. Surfel Generation

We denote the true pose of robot at any arbitrary timetby
T(t) = (R(t),t(t))∈SO(3)×R^3. LetTkimu⊂ Wkdenote
the set of timestamps of IMU measurements received within
Wk. After sliding forward the previous time windowWk− 1 ,
we initially estimate robot poses at the timestamp of new
IMU measurementsi.e., {T(t) :t∈Tkimu∩(Wk\Wk− 1 )},
by integrating new accelerometer and gyro measurements.
We then perform linear interpolation between these poses on
so(3)×R^3 to initialise robot poses associated to new lidar
measurements (i.e., those received inWk\Wk− 1 ). For each
new lidar measurement, interpolation is performed between
the two closest (in time) IMU poses. This initial guess is then
used to place raw lidar measurements in the world frame.
Next, we generate surfels (surface elements) by clustering
points based on their positions and timestamps and fitting
ellipsoids to them. First, we divide the space into a set of
cube-shaped cells (voxels) and cluster lidar points withineach
cell and with proximal timestamps together. We then fit an
ellipsoid to each sufficiently large cluster of points basedon a
predetermined threshold. The center of each ellipsoid (position

```
of surfel) and its shape are determined by the sample mean
and covariance of 3D points in the cluster, respectively. Recall
that lengths of an ellipsoid’s principal semi-axes are given
by the reciprocals of the square root of the eigenvalues of
the corresponding covariance matrix. Therefore, the larger the
gap between the two smallest eigenvalues of the covariance
matrix is (normalised by the trace), the “more planar” the
corresponding ellipsoid. We thus quantify planarity of a surfel
by computing a score based on the spectrum of its covariance
matrix [22, Eq. 4] and only keep surfels that are sufficiently
planar. Further, we use the eigenvector corresponding to the
smallest eigenvalue of the covariance matrix to estimate the
surface normal. Finally, we employ a multi-resolution scheme
where the clustering and surfel extraction steps are repeated
at multiple spatial resolutions (voxel sizes).
```
```
B. Continuous-Time Trajectory Optimisation
After generating surfels from new lidar points in Wk,
Wildcat odometry performs a fixed number ofouteriterations
by alternating between (i) matching surfels given the current
trajectory estimate, and (ii) optimising the robot’s trajectory
using IMU measurements and matched surfels and updating
surfels’ positions, which are described in Sections IV-B1 and
IV-B2, respectively. Fig. 3 depicts the procedure of trajectory
optimisation in Wildcat.
1) Surfel Correspondence Update:Each surfel is described
by its estimated position in the world frame, normal vector and
resolution. This information is used to establish correspon-
dences between surfels. Specifically, we conduct k-nearest
neighbour search in the 7 -dimensional descriptor space for
surfels created within the current time windowWkand keep
reciprocal matches whose average timestamps are farther than
a small predefined threshold. We denote the set of matched
surfel pairs byM.
2) Trajectory Update:
Step 2.1 - Updating Sample Poses:Let{ti}ni=1be the set
ofnequidistant timestamps sampled fromWk, andTˆ(ti) =
(Rˆ(ti),ˆt(ti))denote the estimate of robot trajectory at the
sampling timeti. In this step, we first aim to compute small
pose corrections{Tcori }ni=1={(Rcori ,tcori )}ni=1∈(SO(3)×
R^3 )nto update our estimate of robot’s trajectory at sampling
times according to(Rˆ(ti),ˆt(ti))←(Rcori Rˆ(ti),tcori +ˆt(ti)).
We obtain these correction poses by solving the following
local optimisation problem,
```
```
minimise
Tcori∈SO(3)×R^3
i=1,...,n
```
#### ∑

```
t∈Tkimu
```
```
fimut +
```
#### ∑

```
(s,s′)∈M
```
```
fs,s
```
```
′
match. (2)
```
```
The cost functionsfimut andfs,s
```
```
′
match(associated to IMU mea-
surements and matched surfel pairs, respectively) are functions
of (small) subsets of {Tcori }ni=1 and their arguments are
omitted for notation simplicity. Before defining these cost
functions in Section IV-C, we describe key steps taken by
Wildcat odometry to update its estimate of robot’s trajectory.
To solve (2), we use a retraction to formulate the problem as
an optimisation problem over(so(3)×R^3 )n∼=(R^6 )n. This
allows us to express the constrained optimisation problem in
(2) as an unconstrained problem whose decision variables are
```

```
Wildcat Odometry
```
```
Locally optimised trajectory
and surfel map
```
```
Wildcat Pose Graph
```
```
Odometry edges between
consecutive nodes
```
```
Edges between
non-consecutive nodes
```
```
Synchronisation of frame dadabases via p2p
communication between agents
```
Fig. 2: Wildcat consists of two major modules: Wildcat odometry in which IMU and lidar measurements are integrated in a sliding-
window fashion to continuously estimate the robot trajectory and produce submaps for pose-graph optimisation; Pose-graph optimisation
where submaps are used to generate odometry edges as well as loop closures to efficiently map large-scale environments while correcting
the odometry drift. Additionally, Wildcat can be used in multi-agent scenarios by sharing submaps via peer-to-peer (p2p) communication
between the agents and optimising the collecting pose graph.

```
Low resolution High resolution
```
Fig. 3: Continuous-Time optimisation framework in Wildcat.Local
Optimisationis based on multiple cost terms mainly from IMU mea-
surements and surfel correspondences. Local optimisationestimates
a set of discrete poses (Update Sample Poses), then inB-spline
Interpolations, by fitting a cubic B-spline over the corrected samples
and the robot poses obtained from the IMU poses at the sample
times, the robot poses (Update IMU Poses) are updated at a high
temporal resolution. This process is repeated iterativelyto obtain an
accurate estimate of robot trajectory and remove distortion from the
surfel map.

{(rcori ,tcori )}ni=1∈(R^3 ×R^3 )nsuch thatRcori = exp((rcori )∧).
As we see shortly, this optimisation problem is a standard non-
linear least squares problem which we solve (approximately)
using Gauss-Newton. Specifically, we linearise the residuals
and solve the normal equations to obtain{(rcori ,tcori )}ni=1. We
then update our estimate of robot pose at sampling times ac-
cording to(Rˆ(ti),ˆt(ti))←(exp((rcori )∧)Rˆ(ti),tcori +ˆt(ti))
fori∈[n]. We make this process robust to outliers using an
iteratively reweighted least squares (IRLS) scheme based on
the Cauchy M-estimator [27].

Step 2.2 - Updating IMU Poses:To be able to solve the
above optimisation problem sufficiently fast, the number of
sample posesnis typically an order of magnitude smaller than
the number of IMU measurements inWk. However, Wildcat
odometry must maintain an estimate of robot’s trajectory at
a higher rate (say, 100 Hz) to accurately place surfels in
the world frame (see Step 2.3) and also for defining the
cost functions in (2) (see Section IV-C). We therefore use
our corrected sample poses to update our estimate of robot’s
trajectory at IMU timestamps,i.e., {Tˆ(t) : t ∈ Tkimu}.
Since the timestamps of IMU measurements are not aligned
with those of sample poses, we first use a cubic B-spline
interpolation between corrected sample poses{Tˆ(ti)}ni=1to

```
obtain a continuous-time estimate of robot trajectoryi.e.,Tˆsp:
Wk →SE(3)such thatTˆsp(t)denotes the estimated pose
at timet∈ Wk. We then perform another cubic B-spline
interpolation, this time between poses in{T ̆(ti)}ni=1where
T ̆(ti) is an estimate of robot pose at time ti obtained
by linearly interpolating our latest estimate of robot pose
at the two closest timestamps inTkimu. This gives another
continuous-time estimate of robot’s trajectory that we denote
byT ̆sp:Wk→SE(3). We now update our estimate of robot’s
trajectory at IMU timestamps according to^2
```
```
Tˆ(t)←Tˆsp(t)·(T ̆sp(t))−^1 ·Tˆ(t), ∀t∈Tkimu. (3)
```
```
Step 2.3 - Updating Surfel Positions:The surfels’ positions
in the world frame are determined by the estimated robot
trajectory. We use the updated estimate of the robot’s tra-
jectory at IMU timestamps{Tˆ(t) :t∈ Tkimu}to reproject
surfels generated inWk in the world frame. Note that this
step may result in a new set of surfel correspondences in the
next correspondence step (Section IV-B1).
```
```
C. Cost Functions
In this section, we introduce the cost functions used in
(2) for optimising robot’s trajectory. Recall that our optimisa-
tion variables are pose corrections{(Rcori ,tcori )}ni=1computed
to correct the estimated robot poses at thesampling times
{ti}ni=1. The sampling times, however, are not aligned with
surfel or IMU timestamps. As shown in Fig. 4, we use linear
interpolation (on so(3)×R^3 ) to relate the IMU measure-
ments and estimated surfels’ positions to correction poses.
Specifically, consider an arbitrary timeτ ∈ Wkwhereτis
not necessarily in{ti}ni=1. Letta, tb∈ {ti}ni=1be the two
closest sampling times toτsuch thatta≤τ≤tband define
ατ ,(τ−ta)/(tb−ta). We then denote theinterpolated
correction poseat timeτby(R ̄corτ, ̄tcorτ)∈SO(3)×R^3 where
```
```
R ̄corτ ,RotInterpolate(rcora,rcorb , ατ), (4)
̄tcorτ ,LinInterpolate(tcora ,tcorb , ατ). (5)
```
(^2) In (3),Tˆ(t),Tˆsp(t), andT ̆sp(t)are meant to be seen as elements of
SE(3)(using the natural identification between elements ofSO(3)×R^3 and
SE(3)).


HereLinInterpolatedenotes the linear interpolation operator
(1), and RotInterpolate interpolates rotations in a similar
fashion.^3 With this notation, we are now ready to describe
the cost functions below.

Surfel-Matching Cost Functions:Consider a pair of matched
surfels(s, s′) ∈ M. We use interpolated correction poses
at surfel times τs andτs′ to formulate a point-to-plane-
type cost function [28] that penalises misalignment between
sands′ afterapplying correction poses. The eigenvector
corresponding to the smallest eigenvalue of the combined
sample covariance matrices is used as our estimate for the
normal vectorns,s′to the planar patch captured insands′
(see Section IV-A). Our point-to-plane-type cost functionis
defined as

```
fs,s
```
```
′
match,ws,s′
```
#### 〈

```
ns,s′,R ̄corτs′pˆs′+ ̄tcorτs′−R ̄corτspˆs− ̄tcorτs
```
#### 〉 2

#### , (6)

where,ˆpsandpˆs′denote the current estimate of the positions
of surfelssands′, respectively, andws,s′, 1 /(σ^2 +λs,s

′
min)
is a scalar weight defined using the lidar noise varianceσ^2
and the smallest eigenvalueλs,s

′
minof the combined covariance
matrix which quantifies the thickness of the combined surfels.

IMU Cost Functions:Letaτandωτbe the linear accelera-
tion and angular velocity measured by IMU at timeτ∈Tkimu.
These measurements are modelled as,

```
aτ=R(τ)⊤(wa(τ)−g) +ba(τ) +ǫa(τ), (7)
ωτ=ω(τ) +bω(τ) +ǫω(τ), (8)
```
where, (i)wa(τ),ω(τ)∈R^3 denote the true linear accelera-
tion of body in the world frame and the angular velocity of
body relative to the world frame expresses in the body frame,
respectively; (ii) IMU biases are denoted byba(τ),bω(τ)∈
R^3 ; (iii)ǫa(τ)andǫω(τ)are white Gaussian noises; and (iv)
gis the gravity vector in the world frame.

Now consider the IMU measurement received at timeτ∈
Tkimuand letτ 1 , τ 2 be the timestamps of the two subsequent
IMU measurements. We haveτ 2 ≈τ 1 + ∆timu≈τ+ 2∆timu
where∆timuis the (nominal) time difference between subse-
quent IMU measurements (in our case,∆timu= 0. 01 s). The
IMU cost function corresponding to measurements collected
atτ∈Tkimucan be written asfimuτ =faτ+fωτ+fbiasτ where,

```
faτ,‖ωτ−ωˆ(τ)−bω‖^2 Σ−a 1 , (9)
```
```
fωτ,
```
#### ∥

#### ∥

```
∥R ̃(τ)(aτ−ba)−wˆa(τ) +g
```
#### ∥

#### ∥

#### ∥

```
2
Σ−ω^1
```
#### , (10)

```
fbiasτ ,‖bω−ˆbω‖^2 Σ− 1
bω
+‖ba−ˆba‖^2 Σ− 1
ba
```
#### . (11)

where,Σa,Σω,Σba, andΣbω are measurement and biases
covaraince matrices,ˆbωandˆbaare the latest estimates of IMU
biases, andωˆ(τ)andwˆa(τ)are estimates ofω(τ)andwa(τ)

(^3) Interpolating rotations can be done in multiple ways such using spherical
linear interpolation (Slerp) for unit quaternions or by following a geodesic
onSO(3)between the two rotation matrices for duration ofατ. Our current
implementation uses an approximation of Slerp.
Time
IMU
Sample

## t 1

## τs

## Surfel s

## t 0 ti ti+1 ti+

## τ τ 1 τ 2

...
Fig. 4: A schematic of Wildcat data fusion procedure. The sample
poses, whose timestamps (ti) fall between IMU timestamps, are
initialised with linear interpolation between two closestIMU poses.
Similarly, surfels,e.g.,s, are initialised based on linear interpolation
between IMU poses.

```
afterapplying correction poses using the Euler’s method,
```
```
ωˆ(τ),
```
#### 1

```
∆timu
```
#### [

```
log
```
#### (

```
R ̃(τ)⊤R ̃(τ 1 )
```
#### )]∨

#### (12)

```
wˆa(τ),
```
#### 1

```
∆timu^2
```
#### (

```
̃t(τ 2 )− 2 ̃t(τ 1 ) + ̃t(τ)
```
#### )

#### , (13)

```
in which,R ̃(t) =R ̄cort Rˆ(t)and ̃t(t) = ̄tcort +ˆt(t)fort∈
{τ, τ 1 , τ 2 }describe robot pose at timetafter applying the
interpolated correction poses; see (4) and (5).
```
#### V. WILDCATPOSE-GRAPHOPTIMISATION

```
In this section, we describe key components of Wildcat’s
pose-graph optimisation (PGO) module. Wildcat’s odometry
module estimates robot’s trajectory only usinglocalinforma-
tion and thus inevitably suffers from accumulation of error
over time. The PGO module addresses this issue by optimising
trajectory usingglobalinformation, albeit at a lower temporal
resolution.
```
```
A. Submap Generation
The building blocks of our PGO module are submaps.
Submaps encapsulate data over a shortfixed-length time
window. Specifically, each submap is a six-second bundle
of odometry estimates, accumulated local surfel map, and
an estimate of the direction of the gravity vector in the
submap’s local coordinate frame.^4 Wildcat generates submaps
periodically (e.g., every five seconds) after the odometry
module fully processes the corresponding time interval. The
error accumulated within a submap is negligible because
each submap’s internal structure is already optimised by the
odometry’s sliding-window optimisation scheme. This allows
the PGO module to treat each submap as arigidblock whose
configuration in the world coordinate frame can be represented
by that of itslocalcoordinate frame.
In multi-agent collaborative SLAM scenarios, each agent
synchronises its database of submaps (containing submaps
generated by itself and others) with other agents (i.e., other
robots or the base station) within its communication range
via peer-to-peer communication. We refer the reader to [6]
for additional information about our ROS-based data sharing
system, Mule. The maximum size of each submap with a
lidar range of 100 m is about 500 KB, whereas the average
```
(^4) This estimate is obtained using accelerometer measurements and the
trajectory estimated by odometry; see (7).


submap size in underground SubT events was about 100-
KB. [6]. Therefore, Wildcat can easily share submaps between
the agents with a modest communication bandwidth.

B. Pose Graph

Recall that nodes in a pose graph represent (unknown) poses
and edges represent relative noisy rigid-body transformations
between the corresponding pairs of poses.
1) Nodes:The nodes in our pose graph initially correspond
to Wildcat submaps. More specifically, each node represents
the pose of a submap’s local coordinate frame with respect to
the world coordinate frame. Upon adding a new edge to the
pose graph (see below), the PGO module merges nodes whose
corresponding local surfel maps have significant overlap and
whose Mahalanobis distance is below a threshold relative toa
single node. By merging redundant nodes, the computational
cost of our PGO module grows with the size of explored
environment rather than mission duration.
2) Edges:There are two types of edges in a pose graph,
namely odometry and loop-closure edges. The odometry edges
connect consecutive nodes and are obtained from the odometry
module’s estimate of relative rigid-body transformation be-
tween the corresponding two nodes. By contrast, loop-closure
edges (typically) link non-consecutive nodes and are computed
by aligning the local maps of the corresponding pairs of nodes.
If the overlap between the corresponding submaps is sufficient
and if the uncertainty associated to their relative pose is below
a threshold, we use point-to-plane Iterative Closest Point(ICP)
to align the surfel submaps. Otherwise, we first obtain a rough
alignment using global methods such as [29] and use that to
initialise ICP.
Potential loop closure candidates are detected either based
on a Mahalanobis distance search radius or by using existing
place recognition methods. The PGO modular design allows
us to easily integrate place recognition techniques such as
Scan Context [15] with Wildcat. In either case, the loop
closure candidate is added to the pose graph when it passes a
gating test based on the Mahalanobis distance.

C. Optimisation

We denote the pose graph withG= (V,E)whereV= [m]
represents pose graph nodes andEis the pose graph edge set.
LetTi= (Ri,ti)∈SO(3)×R^3 denote the pose of theith
pose graph node in the world coordinate frame. The standard
cost function minimised by pose-graph optimisation methods
can be written as

```
fpgo(T 1 ,... ,Tm) =
```
#### ∑

```
(i,j)∈E
```
```
fij(Ti,Tj), (14)
```
wherefij(Ti,Tj)is the standard squared error residual for
the relative rigid-body transformation betweenTiandTj.
We add an extra term to the standard PGO cost function (14)
to leverage information available about the vertical direction
(direction of gravity in the world frame) through accelerometer
measurements. Specifically, our PGO module minimises the
following cost function,

```
fpgo(T 1 ,... ,Tm) +fup(R 1 ,... ,Rm), (15)
```
```
in which,
```
```
fup(R 1 ,... ,Rm),
```
#### ∑

```
i∈V
```
```
‖Riuˆi−wu‖^2 (16)
```
```
where,uˆiis the estimated vertical direction (i.e.,‖uˆi‖= 1)
in the local frame of theith node andwu,[0 0 1]⊤is the
vertical direction in the world frame. As we mentioned earlier
in Section V-A,ˆuiis calculated using (7) and odometry mod-
ule’s estimated robot trajectory at IMU timestamps. Similar
to the odometry module, to be robust to outliers (e.g., false-
positive loop closures) the PGO module minimises (15) using
an IRLS scheme based on the Cauchy M-estimator.
```
#### VI. EXPERIMENTS

```
In this section, we experimentally evaluate Wildcat on a
diverse collection of real datasets and compare its results
with two state-of-the-art lidar-inertial SLAM methods, namely
FAST-LIO2 [7] and LIO-SAM [3].
```
```
A. Summary of Datasets
The datasets used in our experimental analysis are as
follows (see also Table I for a summary).
1) DARPA SubT Dataset: This dataset was collected by
Team CSIRO Data61 comprises of two Boston Dynamics Spot
robots and two BIA5 ATR tracked robots at the SubT Final
Event in Louisville Mega Cavern. Each robot was equipped
with a spinning pack, designed and engineered at CSIRO. A
picture of this pack is shown in Fig. 6 (right). The spinning
pack (hereafter referred to asSpinningPack) is composed of
a Velodyne VLP-16, with the measurement rate set to 20
Hz, a 9-DoF 3DM-CV5 IMU measuring angular velocity and
linear acceleration at 100 Hz, and four RGB cameras. In the
SpinningPack, the Velodyne lidar is mounted at an inclined
angle on a servomotor spinning around the sensor’szaxis at
0.5 Hz. The servomotor is designed in a way that the spinning
Velodyne VLP-16 provides 120◦vertical Field of View (FoV).
We use the ground truth point cloud map provided by DARPA
to evaluate Wildcat’s multi-agent mapping accuracy.
2) MulRan Dataset: We use the DCC03 sequence of the
MulRan [8] dataset. This publicly available urban driving
dataset was collected using an Ouster OS1-64 (at 10 Hz with
a range of 120 m) on a vehicle in Daejeon, South Korea. The
length of this sequence is about 5 km. Combining GPS, fiber
optic gyro and SLAM, MulRan provides the vehicle motion
ground truth in 6 DoF at 100 Hz.
3) QCAT Dataset: This in-house dataset, including two
large scale sequences named QCAT (FlatPack) and QCAT
(SpinningPack), has been collected at the Queensland Centre
for Advanced Technologies (QCAT) in Brisbane, Australia.
These sequences were captured by two hand-held perception
packs, a FlatPack and a SpinningPack, both designed at
CSIRO. In contrast to theSpinningPack, described earlier, the
FoV in theFlatPackis equal to the vertical FoV provided by
VLP-16 (i.e., 30◦). Fig. 6 (left) shows a picture of the flat
pack. For a fair comparison betweenFlatPackandSpinning-
Pack, each sequence was collected roughly through the same
path across QCAT with the duration of about 2 hours each at a
```

```
TABLE I: Description of the datasets used for experimental evaluation of Wildcat.
```
```
Dataset Description Ground Truth Lidar Spinning No. of Agents
DARPA SubT Final Event Subterranean Environments DARPA’s Pointlcoud Map VLP-16 ✓ 4
MulRan DCC03 [8] Urban Driving/Outdoor GPS/INS/SLAM OS1-6 4 ✗ 1
QCAT (FlatPack) Hand-held Platform/Indoor/Outdoor Surveyed Targets VLP-16 ✗ 1
QCAT (SpinningPack) Hand-held Platform/Indoor/Outdoor Surveyed Targets VLP-16 ✓ 1
```
Fig. 5: Photos from the QCAT (SpinningPack) dataset indoors and
outdoors. The QCAT dataset includes various types of environments.
The confined areas such as through the tunnel or stair cases challenge
SLAM due to the degeneracy of these areas. In some of the photos,
surveyed targets can be seen.

```
Fig. 6:FlatPack(left) versusSpinningPack(right).
```
walking speed. The traverse distance for each dataset is about
5 km. Fig. 5 shows several photos of the dataset across QCAT.
The QCAT dataset is uniquely diverse and challenging and
due to travelling indoors and outdoors, providing an accurate
pose ground truth is not feasible in such a complex and
diverse environment. Instead, we deployed and surveyed over
60 targets scattered across the site (Fig. 12). This ground truth,
described in Sec. VI-D, enables us to evaluate the mapping
accuracy of SLAM systems.

B. Results on DARPA SubT Final Event

Fig. 7 illustrates Wildcat results for the prize run at the
DARPA SubT Challenge Final Event. In this run, four robots
usingSpinningPacks started from the same area and explored
a perceptually challenging subterranean environment (with
sensing degradation due to dust and smoke). The robots shared
their submaps with other agents via peer-to-peer communica-
tion. As shown in the figure, Wildcat produced a globally
consistent map online which precisely aligns with DARPA’s
ground truth. According to DARPA, Wildcat’s map had “0%

```
Fig. 7:Top: Multi-agent globally optimised Wildcat SLAM map from
the robots deployed during the prize run at the DARPA Subterranean
Challenge Final Event at the Louisville Mega Cavern, KY. Themulti-
agent map, shown in red, blue, green and yellow, was reportedby
DARPA to have “0% deviation” from the surveyed ground truth
(in grey) and cover 91% of the course. Note that the clouds were
decimated and filtered to remove noisy points caused by interference
between lidar sensors at long ranges.Bottom: Point-wise comparison
between the map generated on the fly and the ground truth map
provided by DARPA.
```
```
deviation” from the ground truth. It is worth noting that the
ground truth was generated by spending 100 person-hours
using a survey-grade laser scanner, according to DARPA.
We also conduct our own point-wise comparison between
the Wildcat map and the ground truth. We compare the vox-
elised Wildcat map (target) with a resolution of 40 cm against
the ground truth map (reference) with a higher resolution of
1 cm. After a fine alignment between the two point clouds,
each point in target point cloud is associated to the nearest
point in reference point cloud and the distance between the
```

```
0 0.05 0.1 0.15 0.
Distance error (m)
```
```
0
```
```
2000
```
```
4000
```
```
6000
```
```
8000
```
```
10000
```
```
12000
```
```
14000
```
```
No. of points in target
```
```
<95.8%
```
Fig. 8: Histogram of point-wise comparison between the Wildcat map
generated on-the-fly between multiple agents and the high-resolution
ground truth.

corresponding points are computed. The average distance error
between the corresponding points is 3 cm with the standard
deviation of 5 cm as shown in Fig. 7 (bottom). The histogram,
shown in Fig. 8, also demonstrates that more than 95% of the
corresponding points’ distances are less than 10 cm, which is
consistent with DARPA’s evaluation.

C. Results on MulRan Dataset

We use the MulRan dataset to evaluate the accuracy of
odometry and SLAM trajectory estimates as it provides 6-
DoF ground truth. Our evaluation is based on Relative Pose
Error (RPE) for odometry evaluation and Absolute Pose Error
(APE) for the evaluation of SLAM trajectory. Both metrics are
computed using evo [30].
To run LIO-SAM on MulRan DCC03, we used the de-
fault parameters recommended by their authors. Mainly,
the voxel filter parameters i.e., odometrySurfLeafSize,
mappingCornerLeafSize,mappingSurfLeafSizewere set to
default as suggested for outdoor environments. For the SLAM
result, the loop-closure module was enabled with the fre-
quency set to 1 Hz to regulate loop-closure constraints.
For FAST-LIO2 in which raw lidar data are used for map
registration, all the parameters were set to defaults except
cube_side_length which was set to 5000 to be compatible
with the environment size in DCC03. Also, for a fair compari-
son between the SLAM trajectory results of Wildcat and LIO-
SAM, we disabled the place recognition module in Wildcat
so that the loop-closure detection is done based on a fixed
search radius similar to LIO-SAM. That said, note that the
search radius in Wildcat is based on the Mahalanobis distance,
whereas LIO-SAM uses the Euclidean distance.
Fig. 9 shows the box plots related to RPE for the translation
(a) and rotation (b) components as a function of trajectory
length (varying from 50 m to 500 m). The odometry estimates
for LIO-SAM are obtained by disabling loop-closure detec-
tion. The odometry drift of LIO-SAM is slightly smaller than
Wildcat in both translation and orientation. On average, the
translation drift for Wildcat and LIO-SAM odometry over this

```
(a) RPE w.r.t the translation part
```
```
50 100 150 200 250 300 350 400 450 500
Distance travelled (m)
```
```
RPE (
```
```
deg
```
```
)
```
```
(b) RPE w.r.t the orientation part
```
```
(c) APE w.r.t the translation part (d) APE w.r.t the orientation part
Fig. 9: Error evaluation of odometry and SLAM trajectories over
MulRan DCC03. Since FAST-LIO2 is not designed to deal with drift
by detecting loop-closures, we avoided to include it’s results in APE
evaluation.
```
```
sequence is 2.9% and 2.4%, respectively. The average rotation
error per meter of traversed trajectory for Wildcat and LIO-
SAM odometry is 0.01 deg/m and 0.009 deg/m, respectively.
FAST-LIO2’s accuracy is worse than the other two methods
and its drift grows at a higher rate. On average, the translation
drift of FAST-LIO2 is 6.8%, and its average rotation error per
meter is 0.03 deg/m.
Box plots in Fig. 9 (c) and (d) depict APE for translation
and rotation, respectively. These results show that Wildcat
achieves slightly higher accuracy performance than LIO-SAM
after enabling loop closure. This can be attributed to a number
of differences between the two methods such as Wildcat
PGO’s gravity alignment constraints and its candidate loop-
closure verification steps (see Sec. V). Finally, the estimated
trajectory and map by Wildcat’s PGO module are shown in
Fig. 10.
```
```
D. Results on QCAT Dataset
We manually identified and marked the position of the
centre of a number of surveyed targets in the 3D maps
generated by Wildcat, LIO-SAM, and FAST-LIO2; see Fig.
```
11. These targets are placed across our campus, in various
types of environments including indoor office environments,
outdoor open and forested regions as shown in Fig. 12. This


Fig. 10: Generated Wildcat Map overlaid with the estimated trajec-
tory over MulRan dataset sequence DCC03.

variety enables us to compare the robustness and versatility
of Wildcat with prior state-of-the-art methods. It is worth
noting that the targets 11-21 are located in a 3-storey office
environment accessed via internal stairs and the targets 53-
60 are placed throughout a mock-up tunnel. These sections
present a challenge to lidar SLAM systems due to their
complexity and restricted view.
To accurately estimate the position of the centre of a
target, we used a tool developed by automap [31] to fly
through the maps, find the target, and cast two rays toward it
from sufficiently different viewpoints. This process has several
advantages. First, it prevents selecting points that are not on
the target plane. Secondly, it allows us to estimate position
of the centre of targets even if the 3D map points are not
dense enough around the centre. Once the targets’ centre is
selected in the generated 3D map, we can register these targets
with the survey points (as reference) to compute the distance
error between correspondences. This evaluation process varies
from point cloud to point cloud comparison (as we did in Sec.
VI-B), which does not precisely show map accuracy due to
the nearest neighbour procedure in data association.
To obtain the best results for FAST-LIO2 and LIO-SAM,

Fig. 11: Examples of determining the centre of targets. We select
each target’s centre by casting two rays from two angles of view.
The target’s centre is precisely computed via triangulation.

```
Fig. 12: A bird’s-eye view of the QCAT map, generated from
theFlatPackdataset along with the trajectory estimated by Wildcat,
overlaid with the location of surveyed targets deployed across the
site.
```
```
we tuned the parameters mentioned in Sec. VI-C since these
methods are quite sensitive to the voxel filter parameters
selected for indoor or outdoor scenarios. Hence, the results
reported for these methods hereafter are for the best parame-
ters chosen by observing the behaviour of these methods for
different settings after several runs. On the contrary, Wildcat
uses a single common set of voxel filter parameters for all
datasets analysed in this paper.
Table II and Fig. 13 show point-to-point distance errors
between corresponding pairs of mapped and surveyed (ground
truth) targets after an outlier-robust alignment using theM-
estimator Sample Consensus (MSAC) method [32]. In addi-
tion to the mean and standard deviation (std) of error, TableII
also includes root mean square error (RMSE) computed during
alignment. The total number of targets across the QCAT
dataset is 63. However, only a subset of these targets could
be identified in the maps created by LIO-SAM and FAST-
LIO2 due to the fact that both systems exhibited significant
error through the tunnel; see Fig. 14. Additionally, LIO-SAM
generates very sparse maps compared to Wildcat and FAST-
LIO2, hence some of the targets could not be accurately
located. Therefore, despite their failures, in Table II andFig.
13 for LIO-SAM and FAST-LIO2 we report error statistics
only for those targets which were mapped accurately enough
to be manually identified in the generated 3D maps (see the
“number of targets” column in Table II). Unlike LIO-SAM and
FAST-LIO2 which could not complete the QCAT experiments,
Wildcat performed robustly in all regions and mapped all 63
targets.
As shown in Table II, in the case of the QCATFlat-
Packdataset, Wildcat’s average error is less than half of the
average error of LIO-SAM and FAST-LIO2 (without taking
into account significant errors in the remaining points after
the tunnel). Similarly, on the QCATSpinningPackdataset,
Wildcat’s average error is about 80% and 20% less than that
of LIO-SAM and FAST-LIO2, respectively. Furthermore, in
```

TABLE II: Accuracy evaluation over the estimated maps in theQCAT dataset. Note that in total there are 63 targets. All targets were marked
in Wildcat’s map for evaluation. However, the results of LIO-SAM and FAST-LIO2 are only for subsets of targets that we could detect in
their maps before these approaches failed (mainly in the tunnel).

```
Absolute Error
QCATFlatPack QCATSpinningPack
SLAM mean (m) RMSE (m) std (m) # targets mean (m) RMSE (m) std (m) # targets
LIO-SAM [3] 0.92 1.33 0.97 41 1.69 2.52 1.90 38
FAST-LIO2 [7] 1.09 1.38 0.85 53 0.43 0.64 0.47 53
Wildcat (ours) 0.42 0.46 0.19 63 0.34 0.46 0.31 63
```
```
(a) Wildcat (QCATFlatPack) (b) LIO-SAM (QCATFlatPack) (c) FAST-LIO2 (QCATFlatPack)
```
```
(d) Wildcat (QCATSpinningPack) (e) LIO-SAM (QCATSpinningPack) (f) FAST-LIO2 (QCATSpinningPack)
```
Fig. 13: Error histogram, corresponding to the error statistics reported in Table II, over the QCATFlatPackdataset (top row) versus QCAT
SpinningPackdataset (bottom row) for Wildcat (left), LIO-SAM (middle) and FAST-LIO2 (right). Number of targets being selected in each
method is normalised for better comparison. Note that the results of LIO-SAM and FAST-LIO2 only include a subset of targets before these
methods failed.

Fig. 14: Slippage cases for LIO-SAM (left) and FAST-LIO2 (right)
through the tunnel on the QCATSpinningPackdataset.

both datasets Wildcat has the lowest error standard deviation.
These results indicate that, although Wildcat’s performance
is not dependent on a particular sensor configuration, it can
leverage additional information provided by theSpinning-
Packcompared toFlatPackto achieve better performance.
By contrast, FAST-LIO2 performs poorly in comparison in

```
theFlatPackdataset, and LIO-SAM’s performance even de-
grades in theSpinningPackdataset despite having richer data
in comparison to theFlatPackdataset. Additionally, Fig. 13
shows error histograms normalised by the number of targets
identified in each method’s map. These histograms show
that Wildcat outperforms LIO-SAM and FAST-LIO2 in terms
of the fraction of points whose error is below 0.5 m and
also maximum error in both theFlatPackandSpinningPack
datasets.
```
```
E. Runtime and Memory Analysis
In this part, we report and analyse the runtime and memory
consumption of Wildcat in the QCATSpinningPackdataset.
The results reported here are collected on a laptop with an
Intel Xeon W-10885M CPU.
Fig. 15 shows the runtime of the main optimisation loop
in the odometry module throughout the QCATSpinningPack
dataset. The average runtime is about 63.3 ms (approximately
15 Hz) which shows realtime performance. On the perception
pack’s NVIDIA Jetson AGX Xavier onboard computer, the
```

```
0 50 100 150
runtime (ms)
```
```
Wildcat odometry
```
```
Fig. 15: Odometry runtime over the QCATSpinningPackdataset.
```
```
0 1000 2000 3000 4000 5000 6000 7000
Time (s)
```
```
0
```
```
500
```
```
1000
```
```
1500
```
```
No. of submaps/nodes
```
```
No. of submaps
No. of nodes
```
Fig. 16: Total number of generated submaps versus number of pose
graph nodes over the QCATSpinningPackdataset.

odometry module runs at about 1 to 4 Hz (which is fast enough
for processing the current time window before the next one
arrives).

As we mentioned in Section V, one of the key features of
our PGO module is the detection and merging of redundant
nodes. This enables Wildcat to prevent unnecessary growth
of the size of pose-graph optimisation problem over time.
To demonstrate this, we report the total number of submaps
generated and the number of nodes in the pose graph over time
while running Wildcat online on theSpinningPackdataset.
As shown in Fig. 16, the total number of submaps is 1402
for the entire experiment, whereas at the end only 213 nodes
(obtained by merging submaps) were included in the PGO,
resulting in about 85% reduction of the pose graph size. This
key feature thus enables our PGO module to efficiently operate
in long-duration missions and large-scale environments. Fig.
17 also shows the distribution of all generated submaps (grey
circles) and final set of pose graph nodes (green circles), as
well as their corresponding edges.

Finally, we report the memory consumption of the odom-
etry and PGO modules when running Wildcat online on
the QCATSpinningPackdataset. As shown in Fig. 18, the
odometry memory usage plateaus at the beginning of the run
at about 500 MiB. The PGO module consumes more memory
than odometry as it needs to store the submaps as the robot
explores new areas. However, the total memory consumed by
the PGO module for the entire dataset is less than 3 GiB.
Additionally, note that as the robot revisits previously explored
areas, the memory usage plateaus out due to the fact that
PGO merges nodes (and their surfel submaps), thus allowing
Wildcat to map large-scale environments more efficiently.

```
Fig. 17: Wildcat pose graph generated online over the QCATSpin-
ningPackdataset. PGO effectively merge submaps (grey dots) into
nodes (green dots) to reduce the pose graph. Nodes along withthe
odometry and loop closure edges (green lines) are only used in PGO.
```
#### VII. CONCLUSION ANDFUTUREWORK

```
We presented Wildcat, an online 3D lidar-inertial SLAM
system that estimates the robot’s 6-DoF motion and efficiently
map large-scale environments. Moreover, we demonstrated its
exceptional robustness and versatility over the state of the
art across a wide range of environments and with different
types of sensors (VLP-16 in two configurations and OS1-
64) and on different platforms (legged, tracked, hand-held,
car). Our results indicated that Wildcat outperforms two state-
of-the-art methods, especially in challenging environments
such as tunnels and corridors. The robustness of Wildcat had
also been demonstrated at the SubT Challenge Final Event,
where Wildcat running in a decentralised fashion on four
robots produced the best SLAM results with “0% deviation”
according to DARPA.
As for the future work, we plan to improve the resilience
and accuracy of the Wildcat odometry module across a wider
range of environments and perception systems, incorporate
deep learning-based place recognition approaches such as
our recent work [33] into Wildcat for better loop-closure
detection, and investigate scaling strategies that can strengthen
Wildcat’s applicability for multi-agent deployments in larger
environments with longer time scales.
```
```
Fig. 18: Memory usage allocated for the odometry and PGO modules
of Wildcat over the QCATSpinningPackdataset.
```

#### ACKNOWLEDGEMENT

The authors would like to thank the engineering team in the
Robotics and Autonomous Systems Group, CSIRO Data61 for
their support. We would also like to thank AutoMap^5 for their
help in target selection using AutoMap software. Technology
described herein is the subject of International PCT Patent
Application No.: PCT/AU2021/050871 entitled “Multi-Agent
Map Generation”, filed in the name of Commonwealth Sci-
entific and Industrial Research Organisation on August 09,
2021.

#### REFERENCES

[1] W. Xu and F. Zhang, “FAST-LIO: A Fast, Robust
LiDAR-inertial Odometry Package by Tightly-Coupled
Iterated Kalman Filter,”IEEE Robotics and Automation
Letters, vol. 6, no. 2, pp. 3317–3324, 2021.
[2] M. Bosse, R. Zlot, and P. Flick, “Zebedee: Design of
a Spring-Mounted 3-D Range Sensor with Application
to Mobile Mapping,”IEEE Transactions on Robotics,
vol. 28, no. 5, pp. 1104–1119, 2012.
[3] T. Shan, B. Englot, D. Meyers, W. Wang, C. Ratti,
and D. Rus, “LIO-SAM: Tightly-coupled Lidar Iner-
tial Odometry via Smoothing and Mapping,” in 2020
IEEE/RSJ International Conference on Intelligent Robots
and Systems (IROS). IEEE, 2020, pp. 5135–5142.
[4] J. Behley and C. Stachniss, “Efficient Surfel-Based
SLAM using 3D Laser Range Data in Urban Environ-
ments,” inRobotics: Science and Systems, vol. 2018,
2018.
[5] C. Park, P. Moghadam, S. Kim, A. Elfes, C. Fookes,
and S. Sridharan, “Elastic LiDAR Fusion: Dense Map-
Centric Continuous-Time SLAM,” in2018 IEEE In-
ternational Conference on Robotics and Automation
(ICRA). IEEE, 2018, pp. 1206–1213.
[6] N. Hudson, F. Talbot, M. Cox, J. Williams, T. Hines,
A. Pitt, B. Wood, D. Frousheger, K. Lo Surdo, T. Molnar,
R. Steindl, M. Wildie, I. Sa, N. Kottege, K. Stepanas,
E. Hernandez, G. Catt, W. Docherty, B. Tidd, B. Tam,
S. Murrell, M. Bessell, L. Hanson, L. Tychsen-Smith,
H. Suzuki, L. Overset al., “Heterogeneous Ground
and Air Platforms, Homogeneous Sensing: Team CSIRO
Data61’s Approach to the DARPA Subterranean Chal-
lenge,”Field Robotics, vol. 2, pp. 557–594, 2022.
[7] W. Xu, Y. Cai, D. He, J. Lin, and F. Zhang, “Fast-lio2:
Fast direct lidar-inertial odometry,”IEEE Transactions
on Robotics, pp. 1–21, 2022.
[8] G. Kim, Y.-S. Park, Y. Cho, J. Jeong, and A. Kim,
“MulRan: Multimodal Range Dataset for Urban Place
Recognition,”2020 IEEE International Conference on
Robotics and Automation (ICRA), pp. 6246–6253, 2020.
[9] J. Zhang and S. Singh, “LOAM: Lidar Odometry and
Mapping in Real-time,” in Proceedings of Robotics:
Science and Systems, Berkeley, USA, July 2014.
[10] ——, “Low-drift and real-time lidar odometry and Map-
ping,”Autonomous Robots, vol. 41, pp. 401–416, 2017.

(^5) https://automap.io/
[11] H. Wang, C. Wang, C.-L. Chen, and L. Xie, “F-LOAM:
Fast LiDAR Odometry and Mapping,” in2021 IEEE/RSJ
International Conference on Intelligent Robots and Sys-
tems (IROS). IEEE, 2021, pp. 4390–4396.
[12] H. Ye, Y. Chen, and M. Liu, “Tightly coupled 3d lidar
inertial odometry and mapping,” in2019 International
Conference on Robotics and Automation (ICRA). IEEE,
2019, pp. 3144–3150.
[13] T. Shan and B. Englot, “LeGO-LOAM: Lightweight
and Ground-Optimized LiDAR Odometry and Mapping
on Variable Terrain,” in2018 IEEE/RSJ International
Conference on Intelligent Robots and Systems (IROS).
IEEE, 2018, pp. 4758–4765.
[14] G. Kim and A. Kim, “LT-mapper: A Modular Framework
for LiDAR-based Lifelong Mapping,” arXiv preprint
arXiv:2107.07712, 2021.
[15] ——, “Scan Context: Egocentric Spatial Descriptor for
Place Recognition within 3D Point Cloud Map,” in
Proceedings of the IEEE/RSJ International Conference
on Intelligent Robots and Systems, Madrid, Oct. 2018.
[16] C. Le Gentil, T. Vidal-Calleja, and S. Huang,
“IN2LAMA: INertial Lidar Localisation and MApping,”
in2019 International Conference on Robotics and Au-
tomation (ICRA). IEEE, 2019, pp. 6388–6394.
[17] C. Qin, H. Ye, C. E. Pranata, J. Han, S. Zhang, and
M. Liu, “Lins: A lidar-inertial state estimator for robust
and efficient navigation,” in2020 IEEE International
Conference on Robotics and Automation (ICRA). IEEE,
2020, pp. 8899–8906.
[18] H. Alismail, L. D. Baker, and B. Browning, “Continuous
Trajectory Estimation for 3D SLAM from Actuated Li-
dar,” in2014 IEEE International Conference on Robotics
and Automation (ICRA). IEEE, 2014, pp. 6096–6101.
[19] A. Patron-Perez, S. Lovegrove, and G. Sibley, “A Spline-
Based Trajectory Representation for Sensor Fusion and
Rolling Shutter Cameras,”International Journal of Com-
puter Vision, vol. 113, no. 3, pp. 208–219, 2015.
[20] P. Furgale, J. Rehder, and R. Siegwart, “Unified Tempo-
ral and Spatial Calibration for Multi-sensor Systems,” in
2013 IEEE/RSJ International Conference on Intelligent
Robots and Systems. IEEE, 2013, pp. 1280–1286.
[21] C. Park, P. Moghadam, J. L. Williams, S. Kim, S. Srid-
haran, and C. Fookes, “Elasticity Meets Continuous-
Time: Map-Centric Dense 3D LiDAR SLAM,”IEEE
Transactions on Robotics, vol. 38, no. 2, pp. 978–997,
2022.
[22] M. Bosse and R. Zlot, “Continuous 3D Scan-Matching
with a Spinning 2D Laser,” in2009 IEEE International
Conference on Robotics and Automation. IEEE, 2009,
pp. 4312–4319.
[23] S. Lovegrove, A. Patron-Perez, and G. Sibley, “Spline
Fusion: A continuous-time representation for visual-
inertial fusion with application to rolling shutter cam-
eras,” inBMVC, vol. 2, no. 5, 2013, p. 8.
[24] C. H. Tong, P. Furgale, and T. D. Barfoot, “Gaussian
Process Gauss–Newton for non-parametric simultaneous
localization and mapping,”The International Journal of
Robotics Research, vol. 32, no. 5, pp. 507–525, 2013.


[25] C. H. Tong, S. Anderson, H. Dong, and T. D. Barfoot,
“Pose Interpolation for Laser-based Visual Odometry,”
Journal of Field Robotics, vol. 31, no. 5, pp. 731–757,
2014.
[26] D. Droeschel and S. Behnke, “Efficient Continuous-
Time SLAM for 3D Lidar-Based Online Mapping,” in
2018 IEEE International Conference on Robotics and
Automation (ICRA). IEEE, 2018, pp. 5000–5007.
[27] Z. Zhang, “Parameter estimation techniques: A tutorial
with application to conic fitting,” Image and vision
Computing, vol. 15, no. 1, pp. 59–76, 1997.
[28] A. Segal, D. Haehnel, and S. Thrun, “Generalized-icp.”
inRobotics: science and systems, vol. 2, no. 4. Seattle,
WA, 2009, p. 435.
[29] A. Makadia, A. Patterson, and K. Daniilidis, “Fully
automatic registration of 3d point clouds,” in2006 IEEE

```
Computer Society Conference on Computer Vision and
Pattern Recognition (CVPR’06), vol. 1. IEEE, 2006,
pp. 1297–1304.
[30] M. Grupp, “evo: Python package for
the evaluation of odometry and slam.”
https://github.com/MichaelGrupp/evo, 2017.
[31] “Automap,” https://automap.io, accessed: 2022-02-22.
[32] P. H. Torr and A. Zisserman, “MLESAC: A New Robust
Estimator with Application to Estimating Image Geome-
try,”Computer vision and image understanding, vol. 78,
no. 1, pp. 138–156, 2000.
[33] K. Vidanapathirana, M. Ramezani, P. Moghadam,
S. Sridharan, and C. Fookes, “LoGG3D-Net: Locally
guided global descriptor learning for 3D place recog-
nition,” in2022 International Conference on Robotics
and Automation (ICRA), 2022.
```


