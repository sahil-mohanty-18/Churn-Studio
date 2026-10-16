import {$,esc} from './common.js';
export const palette=['#8b72f1','#b29df8','#d7cdfb','#26b798','#f3b769','#e77e91'];
const charts=[];
export function chart(id,type,labels,values,options={}){
 const canvas=$('#'+id); if(!canvas)return;
 if(!window.Chart){canvas.parentElement.innerHTML=`<div class="chart-fallback"><p>Chart CDN unavailable · data view</p>${labels.map((label,i)=>`<p>${esc(label)}: <b>${Number(values[i]).toFixed(2)}</b></p>`).join('')}</div>`;return;}
 const muted=getComputedStyle(document.documentElement).getPropertyValue('--muted').trim();
 const border=getComputedStyle(document.documentElement).getPropertyValue('--border').trim();
 const line=type==='line';const donut=type==='doughnut';
 const dataset={label:options.label||'Churn rate (%)',data:values,backgroundColor:donut?['#e98b9e','#b5a0ef','#46bfa6']:line?'#8b72f115':options.colors||palette,borderColor:line?'#8b72f1':undefined,borderWidth:line?2:0,borderRadius:5,barPercentage:.55,pointRadius:line?2:0,tension:.3,fill:line};
 const config={type,data:{labels,datasets:[dataset]},options:{responsive:true,maintainAspectRatio:false,cutout:'78%',plugins:{legend:{display:false},tooltip:{padding:12,cornerRadius:8}},scales:donut?{}:{x:{grid:{display:false},ticks:{color:muted,font:{size:9},maxRotation:0}},y:{beginAtZero:true,grid:{color:border,drawTicks:false},border:{display:false},ticks:{color:muted,font:{size:9},padding:10}}},...options.chartOptions}};
 if(options.horizontal){config.options.indexAxis='y';config.options.scales.x.grid={color:border};config.options.scales.y.grid={display:false};}
 const instance=new Chart(canvas,config);charts.push(instance);return instance;
}
export function curve(id,data,label){const c=chart(id,'line',data.x,data.y,{label,chartOptions:{parsing:false,scales:{x:{type:'linear',min:0,max:1,ticks:{font:{size:10}},title:{display:true,text:label==='ROC'?'False positive rate':label==='Precision / Recall'?'Recall':'Predicted probability'}},y:{min:0,max:1,ticks:{font:{size:10}},title:{display:true,text:label==='ROC'?'True positive rate':label==='Precision / Recall'?'Precision':'Observed rate'}}}}});if(c){c.data.datasets[0].data=data.x.map((x,i)=>({x,y:data.y[i]}));c.data.datasets[0].fill=false;c.data.datasets[0].pointRadius=0;c.update();}}
export function refreshCharts(){for(const c of charts){if(c.options.scales){for(const axis of Object.values(c.options.scales)){if(axis.ticks)axis.ticks.color=getComputedStyle(document.documentElement).getPropertyValue('--muted').trim();}}c.update();}}
