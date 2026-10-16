import {$,failure} from './common.js';
import {dashboard} from './dashboard.js';
import {predict,simulator} from './predict.js';
import {customers} from './customers.js';
import {batch} from './batch.js';
import {insights} from './insights.js';
import {refreshCharts} from './charts.js';
$('#theme-toggle').onclick=()=>{const theme=document.documentElement.dataset.theme==='dark'?'light':'dark';document.documentElement.dataset.theme=theme;try{localStorage.setItem('churn-theme',theme);}catch{}refreshCharts();};
$('#menu-toggle').onclick=()=>{const open=$('#sidebar').classList.toggle('open');$('#menu-toggle').setAttribute('aria-expanded',String(open));};
document.addEventListener('keydown',e=>{if(e.key==='Escape'){$('#sidebar').classList.remove('open');$('#menu-toggle').setAttribute('aria-expanded','false');}});
$('#close-dialog').onclick=()=>$('#customer-dialog').close();
try{await ({dashboard,predict,batch,customers,insights,simulator}[document.body.dataset.page])();}catch(error){failure(error);}
