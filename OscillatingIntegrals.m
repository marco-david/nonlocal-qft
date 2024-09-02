(* ::Package:: *)

(* ::Input::Initialization:: *)
(*Put the computation ideas above into a Module*)
ImPartOfCommutator[\[Sigma]_,m_,x_,t_,\[CapitalLambda]_]:=Module[{g,\[Omega],integrand,ThetaIntegral,RadialIntegrand,ImPartIntegrand},g[p_]:=Exp[-1/2*\[Sigma]^2*p^2];
\[Omega][p_]:=Sqrt[p^2+m^2];
integrand[p_,\[Theta]_]:=1/(2\[Pi])^3 g[p]/(2\[Omega][p])*(-2I*Sin[g[p]*\[Omega][p]*t-p*x*Cos[\[Theta]]]);
ThetaIntegral =IntegrateChangeVariables[Inactive[Integrate][integrand[p,\[Theta]]*Sin[\[Theta]],{\[Theta],0,Pi}],u,u==Cos[\[Theta]]];
RadialIntegrand=p^2*(2\[Pi])*(Activate[ThetaIntegral]/. u->Cos[\[Theta]]);
ImPartIntegrand = -I*RadialIntegrand;
(*Print[Plot[ImPartIntegrand*Exp[-p/\[CapitalLambda]],{p,0,10*(\[CapitalLambda])}]];*)
NIntegrate[ImPartIntegrand*Exp[-p/\[CapitalLambda]],{p,0,Infinity}]//Quiet
]


(* ::Input::Initialization:: *)
datadir="mathematica_commutator_data";
m=1;
\[CapitalLambda]=10^6;
intbndry=5;


(* ::Input::Initialization:: *)
\[Sigma] = 0.0;
data=ParallelTable[{x,t,Abs[ImPartOfCommutator[\[Sigma],m,x,t,\[CapitalLambda]]]^2},{x,-intbndry,intbndry,0.2},{t,-intbndry,intbndry,0.2}];
flatData000=Flatten[data,1];


Export[FileNameJoin[{NotebookDirectory[],datadir,"data000.h5"}],flatData000]


(*Import[FileNameJoin[{NotebookDirectory[],datadir,"data000.h5"}],"Dataset1"]*)


(* ::Input::Initialization:: *)
plot=ListPlot3D[flatData000,Mesh->None,ColorFunction->"Rainbow",PlotRange->All,AxesLabel->{"x","t","|[,]|^2"}];
plane1=Graphics3D[{Blue,Opacity[0.3],InfinitePlane[{{-5,-5,-1000},{5,5,1000},{0,0,1000}}]}];
plane2=Graphics3D[{Red,Opacity[0.3],InfinitePlane[{{-5,5,-1000},{5,-5,1000},{0,0,1000}}]}];
plot3D=Show[plot,plane1,plane2];


(* ::Input::Initialization:: *)
\[Sigma] = 0.01;
data=ParallelTable[{x,t,Abs[ImPartOfCommutator[\[Sigma],m,x,t,\[CapitalLambda]]]^2},{x,-intbndry,intbndry,0.1},{t,-intbndry,intbndry,0.1}];
flatData001=Flatten[data,1];


Export[FileNameJoin[{NotebookDirectory[],datadir,"data001.h5"}],flatData001]


(* ::Input::Initialization:: *)
plot=ListPlot3D[flatData001,Mesh->None,ColorFunction->"Rainbow",AxesLabel->{"x","t","|[,]|^2"}];
plane1=Graphics3D[{Blue,Opacity[0.3],InfinitePlane[{{-5,-5,-1000},{5,5,1000},{0,0,1000}}]}];
plane2=Graphics3D[{Red,Opacity[0.3],InfinitePlane[{{-5,5,-1000},{5,-5,1000},{0,0,1000}}]}];
Show[plot,plane1,plane2]


(* ::Input::Initialization:: *)
\[Sigma] = 0.1;
data=ParallelTable[{x,t,Abs[ImPartOfCommutator[\[Sigma],m,x,t,\[CapitalLambda]]]^2},{x,-intbndry,intbndry,0.1},{t,-intbndry,intbndry,0.1}];
flatData010=Flatten[data,1];


Export[FileNameJoin[{NotebookDirectory[],datadir,"data010.h5"}],flatData010]


(* ::Input::Initialization:: *)
plot=ListPlot3D[flatData010,Mesh->None,ColorFunction->"Rainbow",AxesLabel->{"x","t","|[,]|^2"}];
plane1=Graphics3D[{Blue,Opacity[0.3],InfinitePlane[{{-5,-5,-1000},{5,5,1000},{0,0,1000}}]}];
plane2=Graphics3D[{Red,Opacity[0.3],InfinitePlane[{{-5,5,-1000},{5,-5,1000},{0,0,1000}}]}];
Show[plot,plane1,plane2]


(* ::Input::Initialization:: *)
\[Sigma] = 0.5;
data=ParallelTable[{x,t,Abs[ImPartOfCommutator[\[Sigma],m,x,t,\[CapitalLambda]]]^2},{x,-intbndry,intbndry,0.1},{t,-intbndry,intbndry,0.1}];
flatData050=Flatten[data,1];


Export[FileNameJoin[{NotebookDirectory[],datadir,"data050.h5"}],flatData050]


(* ::Input::Initialization:: *)
plot=ListPlot3D[flatData050,Mesh->None,ColorFunction->"Rainbow",AxesLabel->{"x","t","|[,]|^2"}];
plane1=Graphics3D[{Blue,Opacity[0.3],InfinitePlane[{{-5,-5,-1000},{5,5,1000},{0,0,1000}}]}];
plane2=Graphics3D[{Red,Opacity[0.3],InfinitePlane[{{-5,5,-1000},{5,-5,1000},{0,0,1000}}]}];
Show[plot,plane1,plane2]


(* ::Input::Initialization:: *)
intbndry=5;


(* ::Input::Initialization:: *)
\[Sigma] = 1.0;
intbndryx = 20;
dx = 0.5;
intbndryt = 20;
dy=0.5;
data=ParallelTable[{x,t,Abs[ImPartOfCommutator[\[Sigma],m,x,t,\[CapitalLambda]]]^2},{x,-intbndryx,intbndryx,dx},{t,-intbndryt,intbndryt,dy}];
flatData100=Flatten[data,1];


Export[FileNameJoin[{NotebookDirectory[],datadir,"data100.h5"}],flatData100]


(* ::Input::Initialization:: *)
plot=ListPlot3D[flatData100,Mesh->None,ColorFunction->"Rainbow",PlotRange->All,AxesLabel->{"x","t","|[,]|^2"}];
plane1=Graphics3D[{Blue,Opacity[0.3],InfinitePlane[{{-5,-5,-1000},{5,5,1000},{0,0,1000}}]}];
plane2=Graphics3D[{Red,Opacity[0.3],InfinitePlane[{{-5,5,-1000},{5,-5,1000},{0,0,1000}}]}];
Show[plot,plane1,plane2]


(* ::Input::Initialization:: *)
\[Sigma] = 10.0;
intbndry = 100;
data=ParallelTable[{x,t,Abs[ImPartOfCommutator[\[Sigma],m,x,t,\[CapitalLambda]]]^2},{x,-intbndry,intbndry,2.0},{t,-intbndry,intbndry,2.0}];
flatData1000=Flatten[data,1];


Export[FileNameJoin[{NotebookDirectory[],datadir,"data1000.h5"}],flatData1000]


(* ::Input::Initialization:: *)
plot=ListPlot3D[flatData1000,Mesh->None,ColorFunction->"Rainbow",PlotRange->All,AxesLabel->{"x","t","|[,]|^2"}];
plane1=Graphics3D[{Blue,Opacity[0.3],InfinitePlane[{{-5,-5,-1000},{5,5,1000},{0,0,1000}}]}];
plane2=Graphics3D[{Red,Opacity[0.3],InfinitePlane[{{-5,5,-1000},{5,-5,1000},{0,0,1000}}]}];
Show[plot,plane1,plane2]
