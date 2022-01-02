Search.setIndex({docnames:["_examples/pyvbmc_example_1","_examples/pyvbmc_example_2","about_us","api/advanced_docs","api/classes/acquisition_functions","api/classes/function_logger","api/classes/iteration_history","api/classes/options","api/classes/parameter_transformer","api/classes/timer","api/classes/variational_posterior","api/classes/vbmc","api/functions/active_sample","api/functions/create_vbmc_animation","api/functions/decorators","api/functions/entropy","api/functions/get_hpd","api/functions/kde1d","api/options/vbmc_options","index","installation","quickstart"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.todo":2,sphinx:56},filenames:["_examples\\pyvbmc_example_1.ipynb","_examples\\pyvbmc_example_2.ipynb","about_us.rst","api\\advanced_docs.rst","api\\classes\\acquisition_functions.rst","api\\classes\\function_logger.rst","api\\classes\\iteration_history.rst","api\\classes\\options.rst","api\\classes\\parameter_transformer.rst","api\\classes\\timer.rst","api\\classes\\variational_posterior.rst","api\\classes\\vbmc.rst","api\\functions\\active_sample.rst","api\\functions\\create_vbmc_animation.rst","api\\functions\\decorators.rst","api\\functions\\entropy.rst","api\\functions\\get_hpd.rst","api\\functions\\kde1d.rst","api\\options\\vbmc_options.rst","index.rst","installation.rst","quickstart.rst"],objects:{"pyvbmc.acquisition_functions":[[4,1,1,"","AbstractAcqFcn"],[4,1,1,"","AcqFcn"],[4,1,1,"","AcqFcnLog"],[4,1,1,"","AcqFcnNoisy"],[4,1,1,"","AcqFcnVanilla"]],"pyvbmc.acquisition_functions.AbstractAcqFcn":[[4,2,1,"","__call__"],[4,2,1,"","get_info"]],"pyvbmc.decorators":[[14,3,1,"","handle_0D_1D_input"]],"pyvbmc.entropy":[[15,3,1,"","entlb_vbmc"],[15,3,1,"","entmc_vbmc"]],"pyvbmc.function_logger":[[5,1,1,"","FunctionLogger"]],"pyvbmc.function_logger.FunctionLogger":[[5,2,1,"","__call__"],[5,2,1,"","add"],[5,2,1,"","finalize"]],"pyvbmc.parameter_transformer":[[8,1,1,"","ParameterTransformer"]],"pyvbmc.parameter_transformer.ParameterTransformer":[[8,2,1,"","__call__"],[8,2,1,"","inverse"],[8,2,1,"","log_abs_det_jacobian"]],"pyvbmc.stats":[[16,3,1,"","get_hpd"],[17,3,1,"","kde1d"],[15,3,1,"","kldiv_mvn"]],"pyvbmc.timer":[[9,1,1,"","Timer"]],"pyvbmc.timer.Timer":[[9,2,1,"","get_duration"],[9,2,1,"","start_timer"],[9,2,1,"","stop_timer"]],"pyvbmc.variational_posterior":[[10,1,1,"","VariationalPosterior"]],"pyvbmc.variational_posterior.VariationalPosterior":[[10,2,1,"","get_bounds"],[10,2,1,"","get_parameters"],[10,2,1,"","kldiv"],[10,2,1,"","mode"],[10,2,1,"","moments"],[10,2,1,"","mtv"],[10,2,1,"","pdf"],[10,2,1,"","plot"],[10,2,1,"","sample"],[10,2,1,"","set_parameters"]],"pyvbmc.vbmc":[[6,1,1,"","IterationHistory"],[7,1,1,"","Options"],[11,1,1,"","VBMC"],[12,3,1,"","active_sample"],[13,3,1,"","create_vbmc_animation"],[11,3,1,"","optimize_vp"],[11,3,1,"","train_gp"],[11,3,1,"","update_K"]],"pyvbmc.vbmc.IterationHistory":[[6,2,1,"","record"],[6,2,1,"","record_iteration"]],"pyvbmc.vbmc.Options":[[7,2,1,"","eval"],[7,2,1,"","init_from_existing_options"],[7,2,1,"","load_options_file"],[7,2,1,"","validate_option_names"]],"pyvbmc.vbmc.VBMC":[[11,2,1,"","determine_best_vp"],[11,2,1,"","finalboost"],[11,2,1,"","optimize"]],pyvbmc:[[4,0,0,"-","acquisition_functions"]]},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:function"},terms:{"0":[1,5,6,11,15,16,18],"00":[0,1],"001":0,"005":18,"006":1,"01":[0,1,18],"02":1,"02064906":0,"0235966":0,"03":[0,1],"04":1,"046":1,"05":[0,1,18],"06":[0,1],"0625":1,"07":1,"0762":1,"0885":0,"09":1,"0d":14,"1":[5,8,10,11,15,16,17,18,19,21],"10":[0,1,17,18,19],"100":[0,1,17,18],"100000":10,"1000000":10,"101":0,"1024":18,"11":[0,1],"116":1,"12":[0,1,18],"1234":18,"13":[0,1,18],"13397955":0,"14":[1,17],"15":[0,1,10,18],"157":0,"1587":0,"159":1,"16":[0,1],"16384":17,"17":[0,1],"18":1,"183":1,"19":[0,1],"191163":1,"197":0,"1d":[10,14,17],"1e":18,"1e3":18,"1e5":10,"1e6":10,"1st":0,"2":[5,10,11,17,18,19],"20":[0,1,10,18,19],"200":18,"2010":17,"2012":15,"2018":[11,19],"2020":[11,19],"2021":2,"20310931":0,"235":15,"24":0,"242":15,"25":[0,1,11,18],"26":0,"272":0,"277":0,"28":0,"280":0,"2916":17,"2957":17,"29th":15,"2_d":10,"2_i":[],"2_k":10,"2d":[10,14],"2nd":0,"3":[18,19,21],"30":[0,1],"31":[0,11,19],"328":1,"33":[0,11,19],"332":0,"35":[0,1,17],"36":0,"37":0,"375":[0,1],"377":1,"38":[1,17],"3e5":[0,1],"3f":[0,1],"4":[15,18],"40":[0,1],"400":1,"41":0,"41318":0,"418":0,"45":[0,1],"46":[0,1],"473":1,"49":0,"5":[11,17,18],"50":[0,1,18],"500":[1,5,18],"51":1,"53":[0,1],"55":[0,1,17],"57":1,"6":[0,1,18],"60":[0,1,18],"61":[0,1],"64":18,"65":[0,1],"68":[0,1,11],"7":[0,1],"70":[0,1],"714":0,"74":1,"75":[0,1],"78":0,"8":[0,1,10,16,18],"80":[0,1,18],"800":1,"81":0,"82":0,"8211":19,"8213":11,"8222":19,"8223":11,"8232":19,"836":1,"841":1,"8413":0,"85":1,"9":[0,1,18],"90":1,"92045185":0,"926":1,"93":1,"95":1,"98":1,"99":1,"999":1,"abstract":4,"boolean":[],"case":[1,7,11],"class":[4,5,6,7,8,9,10,11,15,21],"default":[0,5,7,8,10,11,14,15,16,17,18],"do":[1,18],"final":[0,1,5,11,18],"float":[5,9,10,11,15,16,17],"function":[0,1,5,10,11,12,14,16,17,18,19],"import":[0,1,2,17,18,21],"int":[0,1,5,6,8,10,11,12,14,15,17],"long":21,"new":[7,10,11,18],"public":20,"return":[0,1,4,5,7,8,9,10,11,12,14,15,16,17,18,21],"static":1,"switch":[1,18],"true":[0,1,10,11,15,18,19,21],"try":18,"var":18,"while":[2,11,18],A:[1,4,5,7,8,9,10,11,14,17],As:0,By:10,For:[0,1,11],If:[10,11,14,21],In:[0,1,10,11,19,21],It:[1,10,11],Not:[0,15],One:[10,17],The:[0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21],There:10,These:[7,10,18],To:[0,1,11],With:1,_0:[0,1],_:[0,1,21],__call__:[4,5,8],_exampl:11,aarno:2,abc:4,about:[4,11],abov:[1,7,18],absolut:8,abstract_acq_fcn:4,abstractacqfcn:4,accept:11,accord:[1,10],accur:0,acerbi:[2,11,19],acq:4,acq_info:4,acqf_vbmc:18,acqfcn:4,acqfcnlog:4,acqfcnnoisi:4,acqfcnvanilla:4,acqhedg:18,acqhedgedecai:18,acqhedgeiterwindow:18,acquisit:[3,12,18],acquisition_funct:4,across:[1,19],act:7,action:[0,1],activ:[2,10,12,18,19],active_sampl:3,activeimportancesamplingboxsampl:18,activeimportancesamplingfessthresh:18,activeimportancesamplingmcmcsampl:18,activeimportancesamplingmcmcthin:18,activeimportancesamplingvpsampl:18,activesamplefessthresh:18,activesamplefullupdatepastwarmup:18,activesamplefullupdatethreshold:18,activesamplegpupd:18,activesamplevpupd:18,activesearchbound:18,activevariationalsampl:18,ad:[1,11,18],adam:18,adapt:18,adaptiveentropyalpha:18,adaptivek:18,add:[2,5],add_trac:1,addit:[1,11,18],adopt:1,advanc:[1,11,19],advancedopt:18,affect:0,afford:11,after:[10,18],again:1,aim:0,al:17,alarm:18,algorithm:[0,1,10,11,18],all:[0,1,7,10],allow:[7,18],almost:10,along:19,alpha:18,alreadi:[10,21],also:[0,1,10,18,19],altern:[0,19],although:0,alwai:[1,18],alwaysrefitvarpost:18,among:11,an:[0,1,2,5,7,10,11,15,19,21],analyt:[10,15,19],analyz:[0,1],ani:0,anim:13,annal:17,anneal:18,annealedgpmean:18,anoth:7,appli:[10,18],approach:1,appropri:[10,11],approx:18,approxim:[0,1,10,11,18,19,21],ar:[0,1,2,7,10,11,15,18,19,21],argpo:14,argument:[7,14,21],aris:19,arrai:[1,8,10,15,18],artifici:[0,1,2,19],arxiv:19,assign:10,assum:[0,1,10,17],atleast_2d:[0,1],attribut:7,automat:[10,11,17,18],autos:1,avg:18,axi:[0,1],b:[17,18],back:[11,18],balanc:10,balanceflag:10,banana:[0,1,19],bandwidth:[17,18],base:[0,4,12,17,18,19],basic:[19,21],basicopt:18,bay:[0,18,19],bayesian:[0,10,11,19],beauti:0,becaus:17,becom:10,been:[2,5,6,7,10],befor:18,begin:[0,1],being:[16,17],below:[1,10,11,18,19,21],benchmark:19,best:[11,18,21],bestev:18,bestfracback:18,bestsafesd:18,better:[0,1,11],between:[0,1,5,10,11,15,18,21],beyond:0,bigk:15,black:19,blei:15,blob:17,blue:[1,10,19],bo:0,bool:[5,10,11,14,15],boost:[11,18],botev:17,both:[10,11,18,19],bottom:0,bound:[8,10,11,15,17,18,19,21],boundedtransform:18,box:[0,1,11,18,19,21],boxsearchfrac:18,bracket:21,brief:1,broad:[0,1],bsd3:19,budget:[11,19],build:1,bw_select:17,cach:[5,18],cache_s:5,cachefrac:18,caches:18,calcul:[0,4],call:[0,4,11,12],callabl:[5,7,11],can:[0,1,5,6,7,10,11,18,19,21],cannot:1,care:0,carlo:[0,10,11,15,18,19],caus:17,ceil:17,cell:[0,1],center:[0,1,2,10,18,19],centr:10,chang:[1,7,18],changed_flag:11,cheat:1,check:[0,7,11,18,19],chengkun:2,choic:[0,1],chosen:[1,10,17],circl:1,cite:19,clarif:17,classmethod:7,clearli:1,close:[10,18,19],closer:0,cma:18,cmae:18,code:[17,19,21],cognit:19,color:10,column:10,com:17,combin:[11,19],come:[7,10],common:[1,10],compar:10,compon:[1,10,11,18],comput:[0,1,10,11,15,17,18,19],computation:19,concaten:17,concern:21,confer:15,consecut:18,consid:[0,1,16],consider:0,constant:[0,11,18,21],constrain:[1,5,8,10],constrainedgpmean:18,constraint:[10,18],constructor:11,contain:[4,7,10,11],context:9,continu:[18,19,21],contour:19,contribut:2,control:10,conveni:6,converg:[0,1,11,18,19],convers:19,convert:1,coordin:[8,10,11],copi:[0,10],corner:[0,10,19],cornerplot:10,correl:18,correspond:[1,10,11,15],could:[0,1],count:[0,1],coupl:21,cov:[0,10],covari:[0,10,15,18],covflag:10,covsamplethresh:18,crazi:1,creat:[10,13],create_vbmc_anim:3,credibl:[0,1],criterion:[11,18],cross:1,current:[1,10,11,18,19],custom:10,d:[0,1,5,8,10,15,16,17,18,19],daniel:17,data:[0,1,8,10,16,17],datapoint:10,dataset:16,debug:18,decai:18,decim:0,decor:3,decreas:18,def:[0,1],default_options_path:7,defin:[0,1,7,8,10,11,21],defini:7,degre:10,delai:18,denot:[0,11],densiti:[0,1,10,11,16,17,18,19,21],depend:0,descent:18,describ:[1,7],descript:7,design:[1,18,19],detail:[0,1,10,11,21],detentropyalpha:18,detentropymind:18,detenttolopt:18,deterior:17,determin:[8,10,18],determine_best_vp:11,determinist:18,develop:[3,17],deviat:[0,1,11,18,21],df:10,dh:15,diagnost:[0,1,18],diagon:[0,10,18],diamond:1,dict:[1,4,6,7,10,11,12],dictionari:[10,11],differ:[0,1,10,18],difficult:1,diffus:17,dimens:[5,8,10,11,16],dimension:[0,1,15,17],direct:8,directli:10,discount:18,discret:17,discuss:[0,1],displai:[1,10,18],distanc:[10,18],distibut:19,distribut:[0,10,19],diverg:[1,10,15,18],divid:18,document:[10,11,21],doe:[0,1,11,17],domain:[0,1],dot:[1,19],doubl:18,doublegp:18,doubt:11,dozen:19,draw:[0,1,10,11,15],drawn:10,du:8,duplic:10,durat:[5,9],dure:[10,11,18],dx:10,e:[0,10,11,17,19,21],each:[0,1,10,11,16,18],easi:19,easili:0,effect:19,effici:[0,2,19],either:10,elbo:[0,1,11,18,21],elbo_sd:[0,1,11,21],elbostart:18,elcbo:[11,18],elcboimproweight:18,elcbomidpoint:18,element:10,els:7,empir:18,empiricalgpprior:18,empti:[7,18],enabl:8,end:[0,1],enhanc:10,entri:5,entropi:[3,18],entropyforceswitch:18,entropyswitch:18,error:[11,21],es:18,especi:1,ess:18,estim:[0,1,5,10,11,15,17,18,19,21],et:17,etc:0,eval:[7,18],evalu:[0,1,5,7,8,10,11,18,19],evaluation_paramet:7,even:10,event:1,everi:18,everyth:7,evid:[0,1,11,19,21],evolut:1,exact:[0,1,19],exactli:[1,10,19],examin:21,exampl:[11,17,21],excel:19,except:18,execut:[],exist:[7,9,11],expect:18,expens:19,explicit:18,exploit:[1,19],explor:0,expon:1,exponenti:1,extend:19,extens:19,extra:[0,1,18],extrem:[17,18],f:[0,1,17],face:1,facecolor:10,factor:[1,10,18,19],fail:[1,11,18],fairli:0,fals:[1,10,11,14,18],familiar:21,far:1,fast:[11,17,18,19],fast_opts_n:11,fcai:[0,1,2,19],fcn:18,few:[0,1,19],fig:[1,10],figsiz:10,figur:[0,1,10,19],file:7,finalboost:11,find:[1,10,11,18,21],finit:[1,5,10,11],finnish:[0,1,2,19],first:[0,1,11,15,18],fit:[0,11,18,19],fitnessshap:18,fixed_point:17,flatten:10,flattenend:10,fluff:[0,1],follow:[0,1,7,10,21],forc:18,form:19,format:[0,1],forward:10,found:11,four:21,fourier:17,frac:[0,10],frac_back:11,fraction:[10,11,18],framework:19,freedom:10,from:[0,1,4,7,8,10,11,12,17,18,19,21],fsd:5,full:[10,11,18],fulli:18,fun:[5,11],fun_evaltim:5,function_logg:[4,5,11,12],functionlogg:[3,4,11,12],fund:[0,1,2,19],funevalsperit:18,funevalstart:18,further:17,fval:[5,18],fval_orig:5,g:[0,8,11,19],gaussflag:10,gaussian:[0,1,10,11,17,18],gaussian_process:[4,11,12],gaussianprocess:[4,11,12],gener:[0,1,2,10,19,21],gershman:15,get:[1,10,16,18,19],get_bound:10,get_dur:9,get_hpd:3,get_info:4,get_paramet:10,gif:13,github:[1,11,17,19],given:[4,6,10,11,17],go:[1,11,18],good:[0,1],gp:[4,10,11,12,18],gp_s_n:11,gphypsampl:18,gpintmeanfun:18,gplengthpriormean:18,gplengthpriorstd:18,gpmeanfun:18,gpquadraticmeanbound:18,gpr:4,gpretrainthreshold:18,gpsamplethin:18,gpsamplewidth:18,gpstochasticsteps:18,gptolopt:18,gptoloptact:18,gptoloptmcmc:18,gptoloptmcmcact:18,gptraininitmethod:18,gptrainninit:18,gptrainninitfin:18,gpyreg:[4,11,12],grad_flag:15,gradflag:10,gradient:[10,15,18,19],graph_object:1,greater:1,green:19,grid:[1,17],grotowski:17,ground:[0,1],group:[2,18],guess:11,guid:0,h:15,ha:[0,2,5,6,7,10,11,17,19],half:[1,19],handel:14,handl:[5,10,14,18],hard:[1,21],have:[0,1,2,6,7,10],heavi:[10,18],heavytailsearchfrac:18,hedg:18,height:1,helsinki:2,henc:10,here:[0,1],heta:[],heteroskedast:5,heurist:18,high:[0,1,11,16,18,21],higher:[0,1],highest:10,highli:0,highlight:10,highlight_data:10,histogram:19,histori:[6,11],hoffman:15,how:[0,1,2,21],howev:1,hpd:18,hpd_frac:16,hpd_rang:16,hpd_x:16,hpd_y:16,hpdfrac:18,hpdsearchfrac:18,html:11,http:[11,17],hundr:0,hyp_dict:11,hyp_n:11,hyperparamet:[0,11,18],hyperprior:18,hyprunweight:18,i:[10,11,17,21],ideal:[11,21],ident:10,idx:5,idx_best:11,ight:[],ignor:[7,18],imag:1,immedi:18,immun:17,immut:7,implement:[11,15,17,19],impos:1,imposs:1,improv:[0,18],includ:[0,1],increas:[1,18],increment:18,incrementalwarpdelai:18,independ:[0,1],index:[5,10,11,18],indic:[10,11,16,18],inequ:15,inf:[0,1,10,11,18,21],infer:[2,11,15,19,21],info:[2,18],inform:[0,1,4,11,19,21],ini:7,init_from_existing_opt:7,initdesign:18,initi:[5,6,7,11,18,21],input:[0,4,5,8,10,11,14,18,19,21],insid:[0,1],inspir:17,instal:19,instanc:[4,7,10,11,12,15],instanti:10,instantli:0,integ:18,integervar:18,integr:18,integrategpmean:18,intellig:[0,1,2,19],interact:1,interest:[0,1,11],interfac:[10,11],intern:[10,15],interpol:18,interquartil:0,interv:[0,1,17],introductori:0,invari:10,invers:8,involv:[0,21],io:11,ipython:1,isfinit:[],item:1,iter:[0,6,11,12,18,19],iteration_histori:[11,12],iterationhistori:[3,11,12],its:[0,5,11],itself:19,j:[15,17],jacobian:8,jacobian_flag:15,jensen:15,joint:[11,18],judg:18,jupyt:[1,11],k:[0,1,10,11,15,18],k_new:11,kde1d:3,kdepi:17,keep:18,kei:[6,7,11],kept:18,kernel:17,key_valu:6,keyword:14,kfunmax:18,kl:[10,18],kldiv:[10,15],klgauss:18,know:[0,1,18],knowledg:1,known:[0,19],kroes:17,kullback:[1,10,15],kwarg:14,kwarmup:18,l:[0,11,19],label:10,lacerbi:11,lack:18,lambda:[0,1,10,15,18],landscap:0,larg:[0,1,19],last:[5,11],later:[0,1],latter:0,lb:[0,1,8,10,11,21],lcb:18,le:[1,10],lead:10,learn:15,least:[7,19],left:[0,15],leibler:[1,10,15],len:15,length:[1,17,18],less:1,level:[5,18],li:2,like:1,likelhood:11,likelihood:[11,18,19],limit:[10,19],line:[7,19,21],linspac:1,list:[6,7,10,14],llfun:[0,1],lml_true:[0,1],load:7,load_options_fil:7,locat:[1,10,18],log2:17,log:[4,5,8,11,15,18,19,21],log_abs_det_jacobian:8,logarithm:[0,10],logflag:10,logger:11,logit:18,logpdf:[0,1],longer:2,look:[0,11],loop:[],lot:[0,1],low:18,lower:[0,1,8,10,11,15,17,18,19,21],lower_bound:[8,11,17],lpriorfun:[0,1],luigi:[2,11],m:15,machin:15,machineri:19,made:2,magnitud:18,mai:[11,19],main:[10,18],mainli:2,major:1,make:[1,10],malasampl:18,manag:0,mani:[11,18],manipul:[10,21],manner:19,map:10,margin:[0,1,10,11,18,19],marker:10,marker_symbol:1,marlon:2,mass:[0,11],master:17,match:7,math:[],mathbb:0,mathbf:[0,1],mathcal:0,matlab:[17,19],matplotlib:10,matric:[],matrix:[0,10,15,18],max:[17,18],max_idx:11,maxfunev:18,maximum:[1,10,18],maxit:18,maxiterstochast:18,maxrepeatedobserv:18,mayb:19,mcmc:18,mean:[0,1,10,11,15,18],meaningfulli:1,measur:[0,1,18],median:0,member:2,menu:1,merg:7,mesh:17,meshgrid:1,method:[6,7,10,11,15,18,19,21],metric:19,midpoint:18,might:11,mikko:2,min:[17,18],minfinalcompon:18,minfunev:18,minim:0,minimum:18,minit:18,miss:10,mixtur:[1,10,11,18],mode:[1,10,11,17,18],model:[11,17,19,21],moder:19,modifi:18,modul:0,moment:[0,10,18],momentsrunweight:18,mont:[0,10,11,15,18,19],more:[0,1,2,11,18,19,21],most:0,mtv:10,mu1:15,mu2:15,mu:15,mu_1:15,mu_k:[10,15],much:1,multidimension:[0,1],multimod:17,multipl:[1,6,10,18],multipli:18,multivari:[10,15,18],must:[5,6,10],mvnsearchfrac:18,n:[0,8,10,16,17,18],n_sampl:[0,1,10],nabla_:15,name:[1,7,9,14],nan:5,ndarrai:[4,5,8,10,11,15,16,17],necessari:0,need:[1,11,19],neg:18,negquad:18,neither:[10,11],neural:[11,19],neurip:[11,19],neurosci:19,never:[10,17],new_opt:7,next:[0,17],nmax:[10,18],nois:[5,11,18,19],noise_flag:5,noiseless:11,noises:18,noiseshap:18,noiseshapingfactor:18,noiseshapingthreshold:18,noisi:[4,11,18,19],non:[0,1,10,18],none:[1,5,7,8,9,10,11,17,18],nonempti:18,nonlinear:[10,18],nonlinearsc:18,nonparametr:15,nor:10,norm:0,normal:[0,10,11,15,18,21],note:[0,1,7,10,11,17],notebook:[0,1,11],notifi:18,notimplementederror:[10,15],now:[0,1,11],np:[0,1,4,5,8,10,11,15,17,18],ns:15,nselbo:18,nselboincr:18,nsent:18,nsentact:18,nsentboost:18,nsentfast:18,nsentfastact:18,nsentfastboost:18,nsentfin:18,nsentfineact:18,nsentfineboost:18,nsgpmax:18,nsgpmaxmain:18,nsgpmaxwarmup:18,nssearch:18,number:[0,1,5,8,10,11,12,15,17,18,19],numer:[0,19],numpi:[0,1,4,5,8,10,11,16,17],object:[0,6,7,10,11,21],observ:[0,1,10,18],obtain:[0,1],occupi:1,off:[1,18],often:[0,19,21],omega:15,onc:7,one:[0,1,5,7,10,17],ones:[0,1],onli:[7,11,18,19,21],onlin:18,open:2,optim:[0,1,10,11,13,17,18,21],optim_st:[4,11,12],optimisticvariationalbound:18,optimize_vp:11,optimtoolbox:18,optimum:1,option:[0,1,5,8,10,11,12,14,15,17],options_path:7,orang:[1,10],order:2,origflag:10,origin:[8,10,16,19],other:[7,10,11,17],otherwis:[0,1,7,10],our:[0,1,19],out:19,outperform:19,output:[0,4,5,10,18,19,21],outputfcn:18,outwarpthreshbas:18,outwarpthreshmult:18,outwarpthreshtol:18,over:[0,11,17,18],overwrit:7,overwritten:7,ower:0,p1:10,p2:10,p:[0,8,17],packag:[0,1,10,19],page:11,pair:[0,1],panel:0,paper:19,paramet:[4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21],parameter_transform:[5,8,10],parametertransform:[3,5,10],parametr:17,part:[0,1],particular:21,particularli:1,pass:[10,11,16],past:[2,11,18],patched_argpo:14,patched_kwarg:14,path:[7,13],pdf:[1,10,15],penal:[10,11],penalti:[10,18],per:[18,19],percentil:11,perform:[0,8,11,18,19],pick:[1,11,18],pipelin:21,place:[0,1,11],plan:19,plausibl:[0,1,8,11,18,21],plausible_lower_bound:[8,11],plausible_upper_bound:[8,11],plb:[0,1,8,11,21],pleas:11,plot:[0,10,18,19],plot_data:10,plot_lb:1,plot_styl:10,plot_ub:1,plot_vp_centr:10,plotli:1,plt:10,plu:1,png:1,point:[4,5,8,10,11,12,16,17,18,19,21],port:19,portfolio:18,portion:16,posit:[1,5,10,14],possibl:[0,10],post_cov:0,post_mean:[0,1],post_mod:1,posterior:[0,1,10,11,12,15,16,18,19,21],potenti:19,power:[17,19],pp:11,ppf:[0,1],practic:[1,10],precis:0,preliminari:18,present:[0,1],preview:1,previou:[1,18],previous:[5,19],print:[0,1],prior:[11,18],prior_mu:0,prior_std:0,prior_tau:1,probabilist:19,probabl:[0,10,11],problem:[0,1,11,17,19],proceed:[0,15,19],process:[10,11,19],prod_:0,profil:18,program:[10,19],progress:[19,20],project:19,properli:[],proportion:[10,18],propos:18,proposalfcn:18,prospect:4,provid:[0,5,10,11,18,21],proxim:[1,11],prune:[11,18],pruningthresholdmultipli:18,pub:[0,1,8,11,21],py:[0,17],pymc3:19,python:[2,17,19],pyvbmc:[0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,20,21],pyvbmc_example_1:11,q:[0,1,10],quadrat:18,quadratur:[0,19],quantil:[0,1],queri:10,quick:19,r:0,rac:[],rais:[5,6,7,10,11,15],rand:18,randn:17,random:[0,1,10,17,18],randomli:10,rang:[0,8,11,16,17,18],rank:[11,18],rank_criterion_flag:11,rankcriterion:18,ravel:1,raw:10,rawflag:10,reach:[11,18],real:[5,19],realist:0,reason:19,recal:0,recenc:11,recommend:1,recomput:18,recomputelcbmax:18,record:6,record_iter:6,red:[1,10],reduc:18,refer:[11,15,17],refin:[1,18],refit:18,regard:1,region:[0,1,11,18,21],regular:18,regulat:18,releas:20,reliabl:[17,18],rememb:1,remov:[1,5],reparameter:10,repeat:18,repeatedacqdiscount:18,replac:10,replic:11,report:0,repres:[0,1,8,10,11,19],represent:10,request:10,requir:[0,1,5,18],rescal:18,research:2,reshap:1,resolut:0,resourc:2,respect:[0,11,16],respons:[6,7],result:[5,8,10,11,21],results_dict:11,retrain:18,retri:18,retrymaxfunev:18,return_scalar:14,revers:10,right:15,rosenbrock:[0,1],rotat:18,roto:18,round:17,row:10,run:[10,11,13,18,21],runtim:18,s:[0,1,2,10,15,18],safe_sd:11,sai:19,same:[1,10,18],sampl:[0,1,4,5,10,11,12,15,17,18,19],sample_count:12,sampleextravpmean:18,sampler:18,save:13,sc:[0,1],scalar:[5,11,14],scale:[1,10,18],scalelowerbound:18,scatter3d:1,scenario:[0,1],scene:1,scipi:[0,1],scroll:1,sd:[0,5,11,18],search:[4,18],searchacqfcn:18,searchcachefrac:18,searchcmaesbest:18,searchcmaesvpinit:18,searchmaxfunev:18,searchoptim:18,second:[5,10,15,18,19],see:[0,1,10,11,17,21],seen:0,select:[17,19],self:10,separ:[0,10,17,18],separatesearchgp:18,seri:[0,1],set:[0,1,7,8,10,11,16,17,18,19],set_paramet:10,setup:21,sever:10,sgdstepsiz:18,shape:[0,5,10,11,16,18],should:[0,5,6,7,8,9,10,11,13,14],show:[0,1,10,19],showscal:1,sigma1:15,sigma2:15,sigma:[10,15,18],sigma_k:10,simpl:0,simpli:10,simplic:0,simul:19,simultan:19,sinc:[0,1,10],singl:[0,1,10],size:[0,1,5,18],skip:18,skipactivesamplingafterwarmup:18,skl:[0,1],slicesampl:18,slow:11,slow_opts_n:11,small:[9,18],smith:17,smooth:[18,19],sn2hpd:11,so:[0,1,10,11],soft:10,softmax:15,softwar:[2,19],solut:[0,1,11,18],some:[0,1,18],someth:[1,19],soon:2,sourc:[2,4,5,6,7,8,9,10,11,12,13,14,15,16,17],space:[1,5,8,10,21],special:18,specif:10,specifi:[0,1,6,7,9,10,11],specifytargetnois:18,sqrt:18,stabil:[1,11,18],stabl:[0,1,11,18],stablegpsampl:18,stablegpvpk:18,stage:[1,18],stai:20,stan:19,standard:[0,1,11,18,21],start:[7,9,10,11,18,19],start_tim:9,stat:[0,1,15,16,17],state:11,statist:[0,2,11,17],std:[0,1,18],step:[0,1,18,21],stepsiz:18,still:[1,20],stochast:[5,18],stochasticoptim:18,stop:[9,18],stop_tim:9,stopwarmupreli:18,stopwarmupthresh:18,store:[6,18],str:[6,7,9,10,13,14],strategi:1,strict:[8,11,18],stricter:18,strictli:1,string:7,struct:18,style:10,sub:18,subsequ:0,substanti:0,success_flag:11,suggest:[1,11],sum:[0,1],sum_:[0,10],summar:1,summari:[0,11],summer:2,supplement:19,support:[1,10,11,19],surfac:1,symmetr:[1,10,18],synthet:0,system:[11,19],t:[0,1,10,15,18],tabl:5,tail:[10,18],take:[0,5,10,18,21],taken:[1,11],target:[0,1,11,16,18,19,21],tbd:[],technic:19,techniqu:[0,17],temper:18,temperatur:18,termin:[0,1,18],test:19,textbf:0,th:[10,11],than:[1,10,19],thank:17,thei:[6,10,11],them:[0,7,14,18],themselv:[],theorem:0,theta:10,theta_bnd:10,thi:[0,1,3,4,6,7,11,12,17,18,19],thin:18,thing:11,thorough:[0,1],those:[10,19],though:1,thousand:0,threhsold:18,threshold:[10,18],through:18,thu:1,time:[5,9],timer:3,titl:10,to_imag:1,tobaben:2,toggl:1,toi:[0,1],tol_con:10,tolboundx:18,tolconloss:18,tolcovweight:18,toler:[10,18],tolfunstochast:18,tolgpnois:18,tolgpvar:18,tolgpvarmcmc:18,tolimprov:18,tollength:18,tolsd:18,tolskl:18,tolstablecount:18,tolstablecountfcn:[0,1],tolstableentropyit:18,tolstableexcptfrac:18,tolstablewarmup:18,tolweight:18,tommyod:17,took:5,toolbox:[18,19],top:[0,1],topmost:11,total:[1,10,18],toward:18,trace:[0,19],train:[1,10,11,16,18],train_gp:11,tranform:10,transflag:10,transform:[5,8,10,14,15,17,18],trasform:10,trial:18,trim:1,truecov:18,truemean:18,trust:11,truth:[0,1],tune:[19,20],tupl:15,tutori:[11,21],tv:10,two:[0,1,10,15,17,18],type:18,typic:[1,21],u:8,ub:[0,1,8,10,11,21],unavail:19,unbound:11,uncertainti:[0,1,4,5,11,18],uncertainty_handling_level:5,uncertaintyhandl:18,unconstrain:[0,5,8,10],und:0,under:[0,19],underli:10,understand:19,undo:18,uniform:[0,17,18],uniformli:1,uninform:1,unit:18,univers:2,unknown:[0,5,18,21],unless:18,unlik:17,unnorm:[0,11,19,21],unus:5,up:[0,1,10,11,17,18,19],updat:[5,11,12,18,20],update_k:11,update_layout:1,updaterandomalpha:18,upper:[0,1,8,10,11,17,18,21],upper_bound:[8,11,17],uppergplengthfactor:18,us:[0,1,7,8,9,10,11,17,18],usag:[1,11,19],user:[0,1,5,7,11,18],user_opt:[1,7,11],useropt:7,val:7,valid:[0,1],validate_option_nam:7,valu:[1,4,5,6,7,8,10,11,16,17,18],valueerror:[5,6,7,10,11],vanilla:4,var_ss:11,varat:18,variabl:[0,1,8,10,11,18],variablemean:18,variableweight:18,varianc:[11,18],variat:[0,1,10,11,12,15,18,19],variational_posterior:[4,10,11,12,15],variationalinitrepo:18,variationalposterior:[3,4,11,12,15,19,21],variationalsampl:18,variou:[0,1,11,21],vastli:19,vbmc:[1,3,4,6,7,9,10,12,13,19,21],vector:[0,5,8,10,11,15,17,21],veri:[0,1,19],versa:8,version:[10,11],vertic:1,via:[0,1,4,10,11,17,18,19,21],vice:8,vicin:1,videnc:0,violat:[10,18],virtual:19,visibl:1,visual:21,vp1:10,vp2:10,vp:[0,1,4,10,11,12,15,18,21],vp_centr:10,vstack:1,w:[15,18],w_k:10,wa:[0,1,17,19],wai:[10,21],want:[1,11,19],warm:[0,1,18],warmup:18,warmupcheckmax:18,warmupkeepthreshold:18,warmupkeepthresholdfalsealarm:18,warmupnoimprothreshold:18,warmupopt:18,warp:18,warpcovreg:18,warpeveryit:18,warpmink:18,warprotocorrthresh:18,warprotosc:18,warptolimprov:18,warptolreli:18,warptolsdbas:18,warptolsdmultipli:18,warpundocheck:18,we:[0,1,2,10,11,17,18,19],weigh:1,weight:[10,18],weight_penalti:10,weight_threshold:10,weightedhypcov:18,weightpenalti:18,well:[0,10,11],what:18,whatev:1,when:[1,7,9,11,18,19],where:[0,8,10,13,17],wherea:0,whether:[5,10,15],which:[0,1,5,6,7,8,9,10,11,17,21],who:2,whose:10,wide:17,width:[1,18],window:18,within:11,without:[0,1,18,19],work:[0,1,11,19,20,21],would:[0,1,10],written:[11,19],wrt:18,x0:[0,1,10,11,18,21],x1:1,x2:1,x:[0,1,5,8,10,11,16,19],x_0:1,x_1:[0,1],x_2:0,x_d:[0,1],xa:1,xaxis_titl:1,xb:1,xmesh:17,xs:[0,1,4],xx:1,y:[1,16],yaxis_titl:1,yet:[10,11,20],you:[0,1,10,18,19,21],your:[0,19,21],yy:1,z:[1,17],zaxis_titl:1,zdravko:17,zero:[0,1,10]},titles:["Example 1: Basic usage","Example 2: Understanding the inputs and the output trace","About us","Advanced documentation","Acquisition Functions","FunctionLogger","IterationHistory","Options","ParameterTransformer","Timer","VariationalPosterior","VBMC","active_sample","create_vbmc_animation","decorators","entropy","get_hpd","kde1d","VBMC options","PyVBMC","Installation","Getting started"],titleterms:{"0":0,"1":[0,1],"2":[0,1],"3":[0,1],"4":[0,1],"5":[0,1],"class":3,"function":[3,4],about:2,acknowledg:[0,1,19],acquisit:4,active_sampl:12,advanc:[3,18],alumni:2,basic:[0,18],bound:[0,1],choos:1,code:[0,1],conclus:[0,1],core:2,create_vbmc_anim:13,decor:14,definit:[0,1],develop:2,document:[3,19],entlb_vbmc:15,entmc_vbmc:15,entropi:15,examin:[0,1],exampl:[0,1,19],full:[0,1],functionlogg:5,get:21,get_hpd:16,goal:0,handle_0d_1d_input:14,i:19,infer:[0,1],initi:[0,1],input:1,instal:20,iter:1,iterationhistori:6,join:2,joint:[0,1],kde1d:17,kldiv_mvn:15,licens:19,likelihood:[0,1],log:[0,1],model:[0,1],option:[3,7,18],output:1,paramet:[0,1],parametertransform:8,plot:1,point:[0,1],prior:[0,1],pyvbmc:19,refer:19,remark:0,result:[0,1],run:[0,1,19],setup:[0,1],should:19,sourc:19,start:[0,1,21],summari:21,team:2,timer:9,trace:1,understand:1,us:[2,19],usag:[0,21],variationalposterior:10,vbmc:[0,11,18],visual:[0,1],what:[0,19]}})