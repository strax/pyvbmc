Search.setIndex({docnames:["_examples/pyvbmc_example_1","_examples/pyvbmc_example_2","api/advanced_docs","api/classes/acquisition_functions","api/classes/function_logger","api/classes/iteration_history","api/classes/options","api/classes/parameter_transformer","api/classes/timer","api/classes/variational_posterior","api/classes/vbmc","api/functions/active_sample","api/functions/create_vbmc_animation","api/functions/decorators","api/functions/entropy","api/functions/get_hpd","api/functions/kde1d","api/options/vbmc_options","index","installation"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.todo":2,sphinx:56},filenames:["_examples/pyvbmc_example_1.ipynb","_examples/pyvbmc_example_2.ipynb","api/advanced_docs.rst","api/classes/acquisition_functions.rst","api/classes/function_logger.rst","api/classes/iteration_history.rst","api/classes/options.rst","api/classes/parameter_transformer.rst","api/classes/timer.rst","api/classes/variational_posterior.rst","api/classes/vbmc.rst","api/functions/active_sample.rst","api/functions/create_vbmc_animation.rst","api/functions/decorators.rst","api/functions/entropy.rst","api/functions/get_hpd.rst","api/functions/kde1d.rst","api/options/vbmc_options.rst","index.rst","installation.rst"],objects:{"pyvbmc.acquisition_functions":[[3,1,1,"","AbstractAcqFcn"],[3,1,1,"","AcqFcn"],[3,1,1,"","AcqFcnLog"],[3,1,1,"","AcqFcnNoisy"],[3,1,1,"","AcqFcnVanilla"]],"pyvbmc.acquisition_functions.AbstractAcqFcn":[[3,2,1,"","__call__"],[3,2,1,"","get_info"]],"pyvbmc.decorators":[[13,3,1,"","handle_0D_1D_input"]],"pyvbmc.entropy":[[14,3,1,"","entlb_vbmc"],[14,3,1,"","entmc_vbmc"]],"pyvbmc.function_logger":[[4,1,1,"","FunctionLogger"]],"pyvbmc.function_logger.FunctionLogger":[[4,2,1,"","__call__"],[4,2,1,"","add"],[4,2,1,"","finalize"]],"pyvbmc.parameter_transformer":[[7,1,1,"","ParameterTransformer"]],"pyvbmc.parameter_transformer.ParameterTransformer":[[7,2,1,"","__call__"],[7,2,1,"","inverse"],[7,2,1,"","log_abs_det_jacobian"]],"pyvbmc.stats":[[15,3,1,"","get_hpd"],[16,3,1,"","kde1d"],[14,3,1,"","kldiv_mvn"]],"pyvbmc.timer":[[8,1,1,"","Timer"]],"pyvbmc.timer.Timer":[[8,2,1,"","get_duration"],[8,2,1,"","start_timer"],[8,2,1,"","stop_timer"]],"pyvbmc.variational_posterior":[[9,1,1,"","VariationalPosterior"]],"pyvbmc.variational_posterior.VariationalPosterior":[[9,2,1,"","get_bounds"],[9,2,1,"","get_parameters"],[9,2,1,"","kldiv"],[9,2,1,"","mode"],[9,2,1,"","moments"],[9,2,1,"","mtv"],[9,2,1,"","pdf"],[9,2,1,"","plot"],[9,2,1,"","sample"],[9,2,1,"","set_parameters"]],"pyvbmc.vbmc":[[5,1,1,"","IterationHistory"],[6,1,1,"","Options"],[10,1,1,"","VBMC"],[11,3,1,"","active_sample"],[12,3,1,"","create_vbmc_animation"],[10,3,1,"","optimize_vp"],[10,3,1,"","train_gp"],[10,3,1,"","update_K"]],"pyvbmc.vbmc.IterationHistory":[[5,2,1,"","record"],[5,2,1,"","record_iteration"]],"pyvbmc.vbmc.Options":[[6,2,1,"","eval"],[6,2,1,"","init_from_existing_options"],[6,2,1,"","load_options_file"],[6,2,1,"","validate_option_names"]],"pyvbmc.vbmc.VBMC":[[10,2,1,"","determine_best_vp"],[10,2,1,"","finalboost"],[10,2,1,"","optimize"]],pyvbmc:[[3,0,0,"-","acquisition_functions"]]},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:function"},terms:{"0":[1,4,5,9,10,14,15],"00":[0,1],"000":0,"005":[],"006":1,"01":[0,1],"02":[0,1],"0298016":[],"03":1,"0311":0,"0352":0,"0367":0,"04":[0,1],"04563237":[],"046":1,"04778539":0,"05":1,"06":[0,1],"0625":1,"0626":[],"0661":[],"07":1,"071":0,"0762":1,"08":0,"08992236":0,"09":1,"0924":[],"09654754":0,"0d":13,"1":[4,7,9,10,14,15,16,18],"10":[0,1,16],"100":[0,1,16],"100000":9,"1000000":9,"101":0,"1024":[],"109":0,"11":[0,1],"115":[],"116":1,"12":[0,1],"1234":[],"12980":[],"13":[0,1],"13501":[],"1392":[],"1393":[],"1394":[],"1395":[],"1396":[],"14":[0,1,16],"1414":[],"1415":[],"1416":[],"1417":[],"1418":[],"146316":0,"147":[],"14782048":[],"15":[0,1,9],"15525083":0,"156":[],"157":[],"158":[],"1587":0,"159":[0,1],"16":1,"160":[],"16384":16,"17":1,"173":[],"174":[],"175":[],"176":[],"177":[],"18":1,"183":1,"19":1,"191163":1,"1d":[9,13,16],"1d653a11e663":[],"1e":[],"1e3":[],"1e5":9,"1e6":9,"1st":0,"2":[4,9,16,18],"20":[0,1,9],"200":[],"2010":16,"2012":14,"2020":10,"21":0,"21264078":[],"22":[],"2231":[],"2232":[],"2233":[],"2234":[],"2235":[],"23":0,"235":14,"242":14,"25":[0,1,10],"26":0,"272":0,"28":0,"285":0,"29":[],"2916":16,"2957":16,"29th":14,"2d":13,"2nd":0,"3":18,"30":[0,1],"31":[],"32":0,"327":0,"328":1,"33":0,"34":[],"35":[0,1,16],"36":0,"368":[],"37":[],"375":1,"377":1,"38":[0,1,16],"39":[],"3e5":[0,1],"3f":[0,1],"4":14,"40":[0,1],"400":1,"41":[],"414":0,"42":[],"426":0,"43":0,"45":[0,1],"46":1,"47":[],"473":1,"5":[10,16],"50":[0,1],"500":[1,4],"51":1,"53":1,"55":[0,1,16],"57":1,"58":0,"59":0,"6":[0,1],"60":[0,1],"606":[],"61":1,"62":[],"64":[],"65":[0,1],"651":[],"68":[0,1,10],"7":[0,1],"70":[0,1],"71":[],"74":1,"75":[0,1],"757":[],"78":0,"79":0,"8":[0,1,9,15],"80":[0,1],"800":1,"80004416":0,"82":0,"834":[],"835":[],"836":1,"837":[],"838":[],"84":0,"841":[0,1],"8413":0,"85":[0,1],"86":[],"87":[],"88":[],"89":[],"9":[0,1],"90":1,"926":1,"93":[0,1],"9314528":[],"95":1,"96":[],"961":[],"962":[],"963":[],"964":[],"965":[],"971":[],"972":[],"973":[],"974":[],"975":[],"98":1,"99":1,"999":1,"abstract":3,"boolean":9,"case":[1,6,10],"class":[3,4,5,6,7,8,9,10,14],"default":[0,4,6,7,9,10,13,14,15,16,17],"do":[1,7,15,17],"final":[0,1,4,10],"float":[4,8,9,10,14,15,16],"function":[0,1,4,5,9,10,11,13,15,16],"import":[0,1,9,16],"int":[0,1,4,5,7,9,10,11,13,14,16],"new":[6,10],"return":[0,1,3,4,6,7,8,9,10,11,13,14,15,16,18],"static":1,"switch":1,"true":[0,1,9,10,14],"try":[],"var":[],"while":10,A:[1,3,4,6,7,8,9,10,13,16],As:0,By:9,For:[0,1],If:[9,10,13],In:[0,1,10],It:[1,10],Not:[0,14],One:[9,16],The:[0,1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],These:[6,17],To:[0,1,10],With:1,_0:[0,1],_:[0,1],__call__:[3,4,7],__compute_nlz:[],__core_comput:[],__gp_obj_fun:[],__init__:10,_boundscheck:10,_check_warmup_end_condit:10,_choleski:[],_compute_reliability_index:10,_create_result_dict:10,_datacopi:[],_estimate_nois:10,_eval_full_elcbo:10,_get_gp_training_opt:10,_get_hpd:10,_get_hyp_cov:10,_get_search_point:11,_get_training_data:10,_gp_hyp:10,_gplogjoint:10,_initialize_full_elcbo:10,_is_finish:10,_is_gp_sampling_finish:10,_negelcbo:10,_recompute_lcbmax:10,_reupdate_gp:10,_robustsamplefromvp:9,_setup_vbmc_after_warmup:10,_siev:10,_soft_bound_loss:10,_vbinit:10,_vp_bound_loss:10,a1:[],abc:3,about:3,abov:[1,6],absolut:7,abstract_acq_fcn:3,abstractacqfcn:3,accord:[1,9],accur:0,acq:3,acq_info:3,acqf_vbmc:3,acqfcn:3,acqfcnlog:3,acqfcnnoisi:3,acqfcnvanila:3,acqfcnvanilla:3,acqflog_vbmc:3,acqfsn2_vbmc:3,acqhedg:[],acqhedge_vbmc:10,acqhedgedecai:[],acqhedgeiterwindow:[],acquisit:[2,11],acquisition_funct:3,acqus_vbmc:3,acqwrapper_vbmc:3,across:1,act:6,action:[0,1],activ:[9,10,11],active_sampl:[2,10],activeimportancesamplingboxsampl:[],activeimportancesamplingfessthresh:[],activeimportancesamplingmcmcsampl:[],activeimportancesamplingmcmcthin:[],activeimportancesamplingvpsampl:[],activesample_vbmc:11,activesamplefessthresh:[],activesamplefullupdatepastwarmup:[],activesamplefullupdatethreshold:[],activesamplegpupd:[],activesamplevpupd:[],activesearchbound:[],activevariationalsampl:[],ad:[1,10],adam:[],adapt:[],adaptiveentropyalpha:[],adaptivek:[],add:4,add_trac:1,addit:[1,10],admiss:10,adopt:1,advanc:[1,18],advancedopt:[],affect:0,afford:10,after:[],again:1,aim:0,al:16,alarm:[],algorithm:[0,1,10,18],all:[0,1,5,6,9],allow:6,alpha:[],alreadi:[9,18],also:[0,1,9,18],altern:[0,18],although:0,alwai:1,alwaysrefitvarpost:[],am:10,among:10,an:[0,1,4,6,9,10,14,18],anaconda3:[],analyt:14,analyz:[0,1],ani:0,anim:12,annal:16,anneal:[],annealedgpmean:[],anoth:[6,9],api:[],appli:[],approach:1,appropri:10,approx:[],approxim:[0,1,9,18],ar:[0,1,6,9,10,14,17,18],argpo:13,argsort:[],argument:[6,13],arrai:[1,7,9,14],artifici:[0,1,18],assign:9,assum:[0,1,9,16],atleast_2d:[0,1],attribut:6,author:[],automat:[10,16],autos:1,avg:[],axi:[0,1],b:16,back:10,bad:18,balanc:9,balanceflag:9,banana:[0,1],bandwidth:16,base:[0,3,11,16,18],basic:[10,18],basicopt:[],bay:[0,18],bayesian:[0,18],beauti:0,becaus:16,becom:9,been:[4,5,6,9,10],befor:[],begin:[0,1],being:[15,16],below:[1,9,10,17],benchmark:18,best:10,best_vbmc:10,bestev:[],bestfracback:[],bestsafesd:[],better:[0,1,10,18],between:[0,1,4,9,14],beyond:0,bigk:14,blei:14,blob:16,blue:[1,9],bo:0,bool:[4,9,10,13,14],boost:10,botev:16,both:[10,17,18],bother:10,bottom:0,bound:[7,9,10,14,16,18],boundedtransform:[],boundscheck_vbmc:10,box:[0,1,10],boxsearchfrac:[],brief:1,broad:[0,1],budget:18,build:1,bw_select:16,c:[],cach:4,cache_s:4,cachefrac:[],caches:[],calcul:[0,3],call:[0,3,10,11],callabl:[4,6,10],calss:8,can:[0,1,4,5,6,9,10,17,18],cannot:1,care:0,carlo:[0,9,14],caus:16,ceil:16,cell:[0,1],center:[0,1,9],centr:9,chang:[1,6,17],changed_flag:10,cheat:1,check:[0,6,10],check_finit:[],choic:[0,1],choleski:[],chosen:[1,9,16],circl:1,clarif:16,classmethod:6,clean:[],clearli:1,close:9,closer:0,cma:[],cmae:[],code:16,cognit:18,color:9,column:9,com:16,combin:10,come:[6,9],common:1,comparison:18,complet:10,compon:[1,9,10],comput:[0,1,9,10,14,16,18],computation:18,compute_grad:[],compute_nlz:[],compute_nlz_grad:[],compute_prior:[],concaten:16,confer:14,consecut:[],consid:[0,1,10,15],consider:0,constant:0,constrain:[1,4,7],constrainedgpmean:[],constraint:9,contain:[3,6,10,18],context:[8,9],continu:[],control:9,conveni:5,converg:[0,1],convert:1,coordin:[7,9,10],copi:[0,9],corner:[0,9],cornerplot:9,correl:[],correspond:[1,9,10,14],could:[0,1,10],couldn:10,count:[0,1],cov:[0,9],covari:[0,9,14],covflag:9,covsamplethresh:[],crazi:1,creat:[8,12],create_vbmc_anim:2,credibl:[0,1],criterion:10,cross:1,current:[1,9],custom:9,d:[0,1,4,7,9,14,15,16],daniel:16,data:[0,1,7,9,15,16],datapoint:9,dataset:15,debug:[],decai:[],decim:0,decomp_choleski:[],decor:2,decreas:[],def:[0,1],default_options_path:6,defin:[0,1,6,7,10],defini:6,degre:9,delai:[],denot:[0,10],densiti:[0,1,9,10,15,16],depend:[0,5],descent:[],describ:[1,6],descript:[6,10],design:[1,18],detail:[0,1],detentropyalpha:[],detentropymind:[],detenttolopt:[],deterior:16,determin:[7,9],determine_best_vp:10,determinist:[],develop:[2,16],deviat:[0,1],df:9,dh:14,diagnost:[0,1],diagon:0,diamond:1,dict:[1,3,5,6,9,10,11],dictionari:[9,10],did:10,differ:[0,1,9],difficult:1,diffus:16,dimens:[4,7,9,10,15],dimension:[0,1,14,16],direct:7,directli:9,discount:[],discret:16,discuss:[0,1,3],displai:1,distanc:9,distibut:9,distribut:[0,9,18],diverg:[1,9,14],divid:17,dnlz:[],docstr:5,document:[],doe:[0,1,10,16],domain:[0,1],dot:1,doubl:[],doublegp:[],doubt:10,draw:[0,1,9,10,14],du:7,duplic:9,durat:[4,8],dure:10,e:[0,10,16,18],each:[0,1,9,10,15],easi:18,easili:0,effici:[0,18],elbo:[0,1,10],elbo_sd:[0,1,10],elbostart:[],elcbo:10,elcboimproweight:[],elcbomidpoint:[],element:[5,9],els:6,empir:[],empiricalgpprior:[],empti:[6,10],enabl:7,end:[0,1],enhanc:9,entlb_vbmc:[],entmc_vbmc:[],entri:4,entropi:2,entropyforceswitch:[],entropyswitch:[],env:[],equal:10,es:[],especi:[1,18],ess:[],estim:[0,1,4,9,10,14,16],et:16,etc:0,eval:6,evalu:[0,1,4,6,7,9,18],evaluation_paramet:6,even:[10,18],event:1,everi:[],everyth:[4,6,7,14],evid:[0,1,18],evolut:1,exact:[0,1],exactli:[1,9],exampl:16,except:[],execut:10,exist:[6,8,10],expect:17,expens:18,experiment:10,explicit:[],exploit:1,explor:0,expon:1,exponenti:1,extens:18,extra:[0,1],extrem:16,f:[0,1,16],f_min_fil:[],face:1,facecolor:9,factor:[1,18],fail:1,fairli:0,fals:[1,9,10,13],familiar:18,far:1,fast:[10,16],fast_opts_n:10,fcai:[0,1],fcn:[],featur:10,few:[0,1],fig:[1,9],figsiz:9,figur:[0,1,9],file:6,fill:10,finalboost:10,finalboost_vbmc:10,find:[1,9,10,17],fine:10,finit:[1,4,10],finnish:[0,1],first:[0,1,14],fit:[0,10,18],fitnessshap:[],fix:[10,15],fixed_point:16,flatten:9,flattenend:9,fluff:[0,1],follow:[0,1,6,9],forc:[],format:[0,1],found:10,fourier:16,frac:0,frac_back:10,fraction:[9,10],freedom:9,from:[0,1,3,6,7,9,10,11,16,18],fsd:4,full:18,fulli:[],fun:[4,10],fun_evaltim:4,function_logg:[3,4,10,11],functionlogg:[2,3,10,11],fund:[0,1],funevalsperit:[],funevalstart:[],funlogger_vbmc:4,further:16,fval:4,fval_orig:4,g:[0,7,10,18],gaussflag:9,gaussian:[0,1,9,10,16],gaussian_process:[3,10,11],gaussian_process_train:[],gaussianprocess:[3,10,11],gener:[0,1,9,10,15,18],gershman:14,get:[1,15],get_bound:9,get_dur:8,get_gptrainopt:10,get_hpd:2,get_info:3,get_lapack_func:[],get_paramet:9,get_traindata_vbmc:10,get_vptheta:9,gethpd_vbmc:[10,15],getsearchpoint:11,gif:12,github:[1,16],given:[3,5,9,10,16],go:[1,10],good:[0,1],gp:[3,9,10,11],gp_s_n:10,gp_sampl:9,gp_train:[],gphypsampl:[],gpintmeanfun:[],gplengthpriormean:[],gplengthpriorstd:[],gplite:10,gplogjoint:10,gpmeanfun:[],gpquadraticmeanbound:[],gpr:3,gpretrainthreshold:[],gpreupdat:10,gpsamplethin:[],gpsamplewidth:[],gpstochasticsteps:[],gptolopt:[],gptoloptact:[],gptoloptmcmc:[],gptoloptmcmcact:[],gptrain:[],gptrain_vbmc:10,gptraininitmethod:[],gptrainninit:[],gptrainninitfin:[],gpyreg:[3,10,11],grad_flag:14,gradflag:9,gradient:[9,14],graph_object:1,greater:1,grid:[1,16],grotowski:16,ground:[0,1],group:17,guess:10,guid:0,h:14,ha:[0,4,5,6,9,10,15,16],half:1,handel:13,handl:[4,13],handle_0d_1d_input:[],hard:[1,10],have:[0,1,5,6,9,10],heavi:9,heavytailsearchfrac:[],hedg:[],height:1,help:10,here:[0,1],heteroskedast:4,heurist:[],high:[0,1,10,15],higher:[0,1],highest:9,highli:0,highlight:9,highlight_data:9,histori:[5,10],hoffman:14,how:[0,1],howev:1,hpd:[],hpd_frac:15,hpd_rang:15,hpd_x:15,hpd_y:15,hpdfrac:[],hpdsearchfrac:[],hprior:[],http:16,hundr:0,hy:[],hyp0:[],hyp:[],hyp_:[],hyp_dict:10,hyp_n:10,hyp_vp:[],hyperparamet:[0,10],hyperprior:[],hyprunweight:[],i:[9,10,16],ident:9,idx:4,idx_best:10,ignor:6,imag:1,immedi:[],immun:16,immut:6,implement:[5,9,10,12,14,16],impos:1,imposs:1,improv:0,includ:[0,1],increas:1,increment:[],incrementalwarpdelai:[],independ:[0,1],index:[4,9,10],indic:[9,10,15],indici:[],inequ:14,inf:[0,1,9,10],infer:[14,18],info:[],inform:[0,1,3,10],ini:6,init:10,init_from_existing_opt:6,init_n:[],initdesign:[],initdesign_vbmc:11,initi:[4,5,6,10],input:[0,3,4,7,9,13,18],insid:[0,1],inspir:16,instal:18,instanc:[3,6,9,10,11,14],instanti:9,instantli:0,instead:18,integ:[],integervar:[],integr:10,integrategpmean:[],intellig:[0,1],interact:1,interest:[0,1],intern:14,interpol:[],interquartil:0,interv:[0,1,16],introductori:0,invers:7,involv:0,ipython:1,isfinit:9,item:1,iter:[0,5,10,11],iteration_histori:[10,11],iterationhistori:[2,10,11],its:[0,4,10,18],j:[14,16],jacobian:7,jacobian_flag:14,jensen:14,joint:[],judg:[],jupyt:1,just:18,k:[0,1,9,10,14],k_new:10,kde1d:2,kdepi:16,keep:[],kei:[5,6,10],kept:[],kernel:16,key_valu:5,keyboardinterrupt:[],keyword:13,kfunmax:[],kl:9,kldiv:[9,14],kldiv_mvn:[],klgauss:[],know:[0,1,17],knowledg:1,known:[0,18],kroes:16,kullback:[1,9,14],kwarg:13,kwarmup:[],l:0,label:9,lacerbi:[],lack:[],lambda:[0,1,9,14],landscap:0,laplac:[],larg:[0,1,18],last:[4,10],later:[0,1,7,10],latter:0,lazili:10,lb:[0,1,7,9,10],lcb:[],le:1,lead:9,learn:14,least:6,left:[0,14],leibler:[1,9,14],len:14,length:[1,16],less:[1,9],level:4,lib:[],licens:[],like:1,likelihood:18,limit:[9,18],linalg:[],linalgerror:[],line:6,linspac:1,list:[5,6,9,10,13],llfun:[0,1],lml_true:[0,1],load:6,load_options_fil:6,locat:1,log2:16,log:[3,4,7,9,10,14,18],log_abs_det_jacobian:7,logarithm:0,logflag:9,logger:10,logit:[],logpdf:[0,1],look:0,loop:10,lot:[0,1],low:[],lower:[0,1,7,9,10,14,16,18],lower_bound:[7,10,16],lpriorfun:[0,1],luigi:[10,15],m:[3,4,5,7,9,10,11,14,15],machin:14,magnitud:[],mai:10,main:[],major:1,make:[1,9],malasampl:[],manag:0,mani:[8,10,17],margin:[0,1,9,18],marker:9,marker_symbol:1,mass:[0,10],master:16,match:6,mathbb:0,mathbf:[0,1],mathcal:0,matlab:[16,18],matplotlib:9,matric:10,matrix:[0,9,14],max:16,max_idx:10,maxfunev:[],maximum:[1,9],maxit:[],maxiterstochast:[],maxrepeatedobserv:[],mayb:18,mcmc:[],mean:[0,1,9,10,14],meaningfulli:1,measur:[0,1],median:0,menu:1,merg:6,mesh:16,meshgrid:1,method:[5,6,14,18],metric:18,midpoint:[],might:10,min:16,minfinalcompon:[],minfunev:[],minim:0,minimum:[],minit:[],minor:[],miss:9,mixtur:[1,9,10],mode:[1,9,16],model:[10,16,18],modifi:[10,17],modul:0,moment:[0,9],momentsrunweight:[],mont:[0,9,14],more:[0,1,9,10,15],most:0,mtv:9,mu1:14,mu2:14,mu:[9,14],mu_1:14,mu_k:14,much:1,multidimension:[0,1],multimod:16,multipl:[1,5,9],multipli:[],multivari:[9,14],must:[4,5,9],mvnkl:14,mvnsearchfrac:[],n:[0,7,9,15,16],n_sampl:[0,1,9],nabla_:14,name:[1,6,8,13],nan:4,ndarrai:[3,4,7,9,10,14,15,16],necessari:0,need:[1,10],neg:[],negelcbo_vbmc:10,negquad:[],neither:[9,10],neurosci:18,never:16,new_opt:6,newli:[8,12],next:[0,16],nlz:[],nmax:9,no_prior:[],nois:[4,10],noise_flag:4,noises:[],noiseshap:[],noiseshapingfactor:[],noiseshapingthreshold:[],noisi:[3,18],non:[0,1],none:[1,4,6,7,8,9,10,11,16],nonempti:[],nonlinear:[],nonlinearsc:[],nonparametr:14,nor:9,norm:0,normal:[0,9,14],note:[0,1,6,10,16,18],notebook:[0,1],noth:15,notifi:[],notimplementederror:[9,14],now:[0,1,9,10],np:[0,1,3,4,7,9,10,14,16],ns:14,ns_gp:[],nselbo:[],nselboincr:[],nsent:[],nsentact:[],nsentboost:[],nsentfast:[],nsentfastact:[],nsentfastboost:[],nsentfin:[],nsentfineact:[],nsentfineboost:[],nsgpmax:[],nsgpmaxmain:[],nsgpmaxwarmup:[],nssearch:[],number:[0,1,4,7,9,10,11,14,16,18],numer:0,numpi:[0,1,3,4,7,9,10,15,16],object:[0,5,6,9],objective_f_1:[],observ:[0,1,9],obtain:[0,1],occupi:1,off:1,often:[0,18],omega:14,onc:6,one:[0,1,4,6,16],ones:[0,1],onli:6,onlin:[],optim:[0,1,9,10,12,16,18],optim_st:[3,10,11],optimisticvariationalbound:[],optimize_vp:10,optimtoolbox:[],optimum:1,option:[0,1,4,7,9,10,11,13,14,16],options_path:6,orang:[1,9],order:[],oriflag:[],origflag:9,origin:[7,9,15,18],other:[3,5,6,9,10,16],otherwis:[0,1,6,9],our:[0,1,18],outperform:18,output:[0,3,4,18],outputfcn:[],outwarpthreshbas:[],outwarpthreshmult:[],outwarpthreshtol:[],over:[0,10,16],overwrit:6,overwrite_a:[],overwritten:6,ower:0,p:[0,7,16],packag:[0,1],page:10,pair:[0,1],panel:0,paper:10,paramet:[3,4,5,6,7,8,9,10,11,12,13,14,15,16,18],parameter_transform:[4,7,9],parametertransform:[2,4,9],parametr:16,part:[0,1,10,11],particularli:1,pass:[9,10,15],past:10,patched_argpo:13,patched_kwarg:13,path:[6,12],pdf:[1,9,14],penal:[9,10],penalti:9,per:15,percentil:10,perform:[0,7,10],pi:[],pick:[1,10],place:[0,1,10],plausibl:[0,1,7,10],plausible_lower_bound:[7,10],plausible_upper_bound:[7,10],plb:[0,1,7,10],pleas:10,plot:[0,9],plot_data:9,plot_lb:1,plot_styl:9,plot_ub:1,plot_vp_centr:9,plotli:1,plt:9,plu:1,png:1,point:[3,4,7,9,11,15,16],port:18,portfolio:[],portion:15,posit:[1,4,9,13],possibl:[0,9],post_cov:0,post_mean:[0,1],post_mod:1,posterior:[0,1,9,10,11,14,15,18],potenti:18,potrf:[],power:16,ppf:[0,1],practic:[1,10],precis:0,preliminari:[],present:[0,1],preview:1,previou:1,previous:4,print:[0,1],prior:10,prior_mu:0,prior_std:0,prior_tau:1,prior_var:0,probabl:[0,9,10],problem:[0,1,16,18],proceed:[0,14],process:[9,10],prod:[],prod_:0,profil:[],program:9,progress:[10,18],project:18,properli:[9,10],proportion:9,propos:[],proposalfcn:[],prospect:3,provid:[0,4,9],proxim:1,prune:10,pruningthresholdmultipli:[],pub:[0,1,7,10],py:[0,10,16],python3:[],python:[12,16,18],pyvbmc:[0,1,3,4,5,6,7,8,9,10,11,12,13,14,15,16],q:[0,1],quadrat:[],quadratur:0,quantil:[0,1],r:0,rais:[4,5,6,9,10,14],rand:[],randn:16,random:[0,1,9,16],randomli:9,rang:[0,7,10,15,16],rank:10,rank_criterion_flag:10,rankcriterion:[],ravel:1,raw:9,rawflag:9,re:[],reach:[],real:[4,18],realist:0,recal:0,recenc:10,recent:[],recommend:1,recomput:[],recompute_lcbmax:10,recomputelcbmax:[],record:5,record_iter:5,red:[1,9],reduc:[],refer:16,refin:1,refit:[],regard:1,region:[0,1,10],regular:[],regulat:[],reliabl:16,rememb:1,remov:[1,4],repeat:[],repeatedacqdiscount:[],replac:[8,9],replic:10,report:0,repositori:18,repres:[0,1,7,10],represent:9,requir:[0,1,4,10,15],rescal:[],rescale_param:9,reshap:1,resolut:0,respect:[0,10,15],respons:[5,6],result:[4,7,9,10],retrain:[],retri:[],retrymaxfunev:[],return_scalar:13,right:14,robustsamplefromvp:9,rosenbrock:[0,1],rotat:7,roto:[],round:16,row:9,run:[9,10,12,18],runtim:[],s2:[],s2_train:[],s:[0,1,9,10,14],s_n:[],safe_sd:10,same:[1,9],sampl:[0,1,3,4,9,10,11,14,16,18],sample_count:11,sampleextravpmean:[],sampler:[],sampler_nam:[],save:12,save_stat:5,sc:[0,1],scalar:[4,10,13],scale:[1,7],scalelowerbound:[],scatter3d:1,scenario:[0,1],scene:1,scipi:[0,1],scroll:1,sd:[0,4,10],se:15,search:3,searchacqfcn:[],searchcachefrac:[],searchcmaesbest:[],searchcmaesvpinit:[],searchmaxfunev:[],searchoptim:[],second:[4,9,14],see:[0,1,9,10,16],seem:10,seen:0,select:[16,18],self:9,separ:[0,16],separatesearchgp:[],seri:[0,1],set:[0,1,6,7,10,15,16,18],set_paramet:9,setupvars_vbmc:10,sgdstepsiz:[],shape:[0,4,9,10,15],should:[0,4,5,6,7,8,9,10,12,13],show:[0,1,18],show_figur:[],showscal:1,sigma1:14,sigma2:14,sigma:[9,14],similar:18,simpl:0,simplic:0,simultan:18,sinc:[0,1],singl:[0,1,9],site:[],size:[0,1,4],skip:[],skipactivesamplingafterwarmup:[],skl:[0,1],slicesampl:[],slow:10,slow_opts_n:10,small:8,smith:16,smooth:[],sn2_div:[],sn2_mat:[],sn2_mult:[],sn2hpd:10,so:[0,1,10],soft:9,softbndloss:10,softmax:14,solut:[0,1,10],some:[0,1],somehow:10,someth:1,sourc:[3,4,5,6,7,8,9,10,11,12,13,14,15,16,18],sp:[],space:[1,4,7,9],special:[],specif:18,specifi:[0,1,5,6,8,9,10],specifytargetnois:[],sqrt:0,stabil:[1,10],stabl:[0,1,10],stablegpsampl:[],stablegpvpk:[],stage:1,standard:[0,1],start:[6,8,9],start_tim:8,stat:[0,1,14,15,16],state:10,statist:[0,10,16],statu:[],std:[0,1],step:[0,1],stepsiz:[],still:1,stochast:4,stochasticoptim:[],stop:8,stop_tim:8,stopwarmupreli:[],stopwarmupthresh:[],store:5,str:[5,6,8,9,12,13],strategi:1,strict:[7,10],stricter:[],strictli:1,string:6,struct:[],style:9,sub:[],subsequ:0,substanti:0,suggest:1,sum:[0,1],sum_:0,summar:1,summari:[0,10],support:[1,9,10],sure:10,surfac:1,swap_sign:[],symmetr:1,synthet:0,t:[0,1,9,10,14],tabl:4,tail:9,take:[0,4,9],taken:[1,10],target:[0,1,10,15],tbd:[8,10,18,19],technic:18,techniqu:[0,16],temper:[],temperatur:[],termin:[0,1,10],test:[10,18],textbf:0,th:[9,10],than:[1,9,18],thank:16,thei:[5,10],them:[0,6,13,17],theorem:0,theta:9,theta_bnd:9,thi:[0,1,2,3,5,6,9,10,11,15,16,18],thin:[],thing:10,thorough:[0,1],those:9,though:1,thousand:0,threhsold:[],threshold:9,through:[],thu:1,tic:8,time:[4,8],timer:2,titl:9,to_imag:1,toc:8,toggl:1,toi:[0,1],tol:[],tol_con:9,tol_opt_mcmc:[],tolboundx:[],tolconloss:[],tolcovweight:[],toler:9,tolfunstochast:[],tolgpnois:[],tolgpvar:[],tolgpvarmcmc:[],tolimprov:[],tollength:[],tolsd:[],tolskl:[],tolstablecount:[],tolstablecountfcn:[0,1],tolstableentropyit:[],tolstableexcptfrac:[],tolstablewarmup:[],tolweight:[],tommyod:16,took:4,toolbox:18,top:[0,1],topmost:10,total:[1,9],toward:[],trace:[0,18],traceback:[],train:[1,9,10,15],train_gp:10,tranform:9,transflag:9,transform:[4,7,9,13,14,16],trial:[],trim:1,truecov:[],truemean:[],truth:[0,1],tune:18,tupl:14,tutori:[],two:[0,1,9,14,16,17],type:17,typic:1,u:7,ub:[0,1,7,9,10],unbound:10,uncertainti:[0,1,3,4,10,18],uncertainty_handling_level:4,uncertaintyhandl:[],unconstrain:[0,4,7,9],und:0,under:0,understand:18,undo:[],uniform:[0,16],uniformli:1,uninform:1,unit:[],unknown:[0,4],unless:17,unlik:16,unnorm:0,untest:10,unus:4,up:[0,1,9,10,16,18],updat:[4,10,11],update_k:10,update_layout:1,updatek:10,updaterandomalpha:[],upper:[0,1,7,9,10,16],upper_bound:[7,10,16],uppergplengthfactor:[],us:[0,1,6,7,8,9,10,16,18],usag:[1,18],user:[0,1,4,6,17],user_opt:[1,6,10],useropt:6,val:6,valid:[0,1],validate_option_nam:6,valu:[1,3,4,5,6,7,9,10,15,16],valueerror:[4,5,6,9,10],vanilla:3,var_ss:10,varat:[],vargrad:10,variabl:[0,1,7,9,10],variablemean:[],variableweight:[],varianc:10,variat:[0,1,9,10,11,14],variational_optim:10,variational_posterior:[3,9,10,11,14],variationalinitrepo:[],variationalposterior:[2,3,10,11,14,18],variationalsampl:[],variou:[0,1,10],vastli:18,vbinit_vbmc:10,vbmc:[1,2,3,5,6,8,9,11,12,18],vbmc_kldiv:9,vbmc_mode:9,vbmc_moment:9,vbmc_mtv:9,vbmc_output:10,vbmc_pdf:9,vbmc_plot:9,vbmc_power:9,vbmc_rnd:9,vbmc_termin:10,vbmc_warmup:10,vector:[0,4,7,9,10,14,16,18],veri:[0,1,18],versa:7,version:9,vertic:1,via:[0,1,3,9,16],vice:7,vicin:1,videnc:0,violat:9,virtual:18,visibl:1,vp2:9,vp:[0,1,3,9,10,11,14],vp_centr:9,vp_repo:10,vpbndloss:10,vpbound:9,vpoptimize_vbmc:10,vpsieve_vbmc:10,vstack:1,w:14,wa:[0,1,10,16,18],wai:9,wait:10,want:[1,10],warm:[0,1],warmup:[],warmupcheckmax:[],warmupkeepthreshold:[],warmupkeepthresholdfalsealarm:[],warmupnoimprothreshold:[],warmupopt:[],warp:10,warpcovreg:[],warpeveryit:[],warpmink:[],warprotocorrthresh:[],warprotosc:[],warptolimprov:[],warptolreli:[],warptolsdbas:[],warptolsdmultipli:[],warpundocheck:[],warpvars_vbmc:7,we:[0,1,9,10,16,17],weigh:1,weight:9,weight_penalti:9,weight_threshold:9,weightedhypcov:[],weightpenalti:[],well:[0,5,9,10],what:[17,18],whatev:1,when:[1,6,8,10],where:[0,7,11,12,16],wherea:0,whether:[4,9,14],which:[0,1,3,4,5,6,7,8,9,10,16,18],whose:9,why:10,wide:16,width:1,window:[],without:[0,1,10],work:[0,1,10,18],would:[0,1,18],written:10,wrt:[],x0:[0,1,9,10],x1:1,x2:1,x:[0,1,4,7,9,10,15,18],x_0:1,x_1:[0,1],x_2:0,x_d:[0,1],x_train:[],xa:1,xaxis_titl:1,xb:1,xmesh:16,xs:[0,1,3],xx:1,y0:[],y:[1,15],y_train:[],yaxis_titl:1,yet:[9,10],you:[0,1,17,18],your:[0,18],yy:1,z:[1,16],zaxis_titl:1,zdravko:16,zero:[0,1,9]},titles:["Example 1: Basic usage","Example 2: Understanding the inputs and the output trace","Advanced documentation","Acquisition Functions","FunctionLogger","IterationHistory","Options","ParameterTransformer","Timer","VariationalPosterior","VBMC","active_sample","create_vbmc_animation","decorators","entropy","get_hpd","kde1d","VBMC options","PyVBMC","Installation"],titleterms:{"0":0,"1":[0,1],"2":[0,1],"3":[0,1],"4":[0,1],"5":[0,1],"class":2,"function":[2,3],acknowledg:[0,1],acquisit:3,active_sampl:11,advanc:[2,17],api:[],attribut:18,author:18,basic:[0,17],bayesian:[],bound:[0,1],carlo:[],choos:1,code:[0,1],conclus:[0,1],create_vbmc_anim:12,decor:13,definit:[0,1],detail:[],document:[2,3,4,5,6,7,8,9,10,14,18],entlb_vbmc:14,entmc_vbmc:14,entropi:14,examin:[0,1],exampl:[0,1,18],full:[0,1],functionlogg:4,get_hpd:15,goal:0,handle_0d_1d_input:13,infer:[0,1],initi:[0,1],input:1,instal:19,iter:1,iterationhistori:5,joint:[0,1],kde1d:16,kldiv_mvn:14,licens:18,likelihood:[0,1],log:[0,1],matlab:[3,4,5,6,7,8,9,10,11,14,15],method:[9,10,15],model:[0,1],mont:[],option:[2,6,17],output:1,paramet:[0,1],parametertransform:7,plot:1,point:[0,1],port:[3,4,5,6,7,8,9,10,15],prior:[0,1],python:[3,4,5,6,7,8,9,10,14],pyvbmc:18,refer:[3,4,5,6,7,8,9,10,11,14,15],remark:0,result:[0,1],run:[0,1],s:[],setup:[0,1],start:[0,1],statu:[3,4,5,6,7,8,9,10,15],timer:8,todo:[3,5,7,8,9,10,15],trace:1,tutori:[],understand:1,usag:0,variat:[],variationalposterior:9,vbmc:[0,10,17],visual:[0,1],welcom:[],what:0}})