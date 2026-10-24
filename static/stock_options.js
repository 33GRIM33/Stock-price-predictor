/* Stock-specific results live only in this browser; no shared result directory. */
(() => {
  const el = id => document.getElementById(id);
  const names = ['Black-Scholes', 'Direct ANN', 'Residual ANN', 'Option LSTM'];
  const colors = ['#64748b', '#2456c4', '#16803c', '#9333ea'];
  const fmt = value => Number(value).toFixed(3);
  let latest = null;
  function table(id, headers, rows, highlighted = []) {
    const table = document.createElement('table'), head = document.createElement('thead');
    const heading = document.createElement('tr');
    headers.forEach(text => { const th = document.createElement('th'); th.scope='col'; th.textContent=text; heading.append(th); });
    head.append(heading); table.append(head);
    const body = document.createElement('tbody');
    rows.forEach((row,index) => { const tr=document.createElement('tr'); if (highlighted.includes(index)) tr.className='best-row'; row.forEach(text => { const td=document.createElement('td'); td.textContent=text; tr.append(td); }); body.append(tr); });
    table.append(body); el(id).replaceChildren(table);
  }
  function svgNode(tag, attrs, text) {
    const node=document.createElementNS('http://www.w3.org/2000/svg',tag);
    Object.entries(attrs).forEach(([key,value]) => node.setAttribute(key,value));
    if (text !== undefined) node.textContent=text;
    return node;
  }
  function errorChart(metrics) {
    const svg=svgNode('svg',{viewBox:'0 0 900 270',role:'img','aria-label':'Held-out RMSE in rupees for each option model'});
    const max=Math.max(...metrics.map(m=>m.rmse),.001);
    metrics.forEach((m,i)=> {
      const width=m.rmse/max*520, y=25+i*52;
      svg.append(svgNode('text',{x:0,y:y+20,'font-size':15},m.model));
      svg.append(svgNode('rect',{x:155,y,width,height:30,fill:colors[names.indexOf(m.model)]}));
      svg.append(svgNode('text',{x:165+width,y:y+20,'font-size':15},`₹${fmt(m.rmse)}`));
    });
    svg.append(svgNode('text',{x:155,y:255,'font-size':14},'RMSE · lower is better'));
    el('option-error-chart').replaceChildren(svg);
  }
  function priceChart(contracts, targetLabel) {
    const rows=[...contracts].sort((a,b)=>a.target-b.target);
    const series=[{name:targetLabel,color:'#111827',values:rows.map(r=>r.target)},
      ...names.map((name,i)=>({name,color:colors[i],values:rows.map(r=>r.predictions[name])}))];
    const max=Math.max(...series.flatMap(s=>s.values),.001);
    const svg=svgNode('svg',{viewBox:'0 0 900 340',role:'img','aria-label':'Target and estimated call premiums sorted by target premium'});
    for (let tick=0;tick<=4;tick++) {
      const y=280-tick*60;
      svg.append(svgNode('line',{x1:75,y1:y,x2:875,y2:y,stroke:'#dce3ec'}));
      svg.append(svgNode('text',{x:0,y:y+5,'font-size':13},`₹${(max*tick/4).toFixed(1)}`));
    }
    series.forEach(s=>svg.append(svgNode('polyline',{points:s.values.map((v,i)=>`${75+i*800/Math.max(rows.length-1,1)},${280-v/max*240}`).join(' '),fill:'none',stroke:s.color,'stroke-width':2,opacity:.85})));
    svg.append(svgNode('text',{x:75,y:320,'font-size':14},`${rows.length} held-out contracts, sorted by target premium (not a time forecast)`));
    el('option-price-chart').replaceChildren(svg);
    el('option-legend').replaceChildren(...series.map(s=> { const span=document.createElement('span'); span.style.color=s.color; span.textContent=`● ${s.name}`; return span; }));
  }
  function render(data) {
    latest=data;
    const observed=data.mode==='observed', target=observed?'Uploaded premium':'Simulated reference';
    el('option-title').textContent=`${data.symbol} · four-model call comparison`;
    el('option-source').textContent=observed?'Uploaded option prices':'Simulated options · based on this stock’s history';
    el('option-context').textContent=`Compared on ${data.split.test.rows} test contracts · ${data.split.test.start} to ${data.split.test.end}`;
    const ranked=[...data.metrics].sort((a,b)=>a.rmse-b.rmse);
    const best=ranked[0], baseline=ranked.find(m=>m.model==='Black-Scholes');
    const winners=ranked.filter(m=>Math.abs(m.rmse-best.rmse)<1e-9);
    el('option-winner').textContent=winners.length>1
      ? `Joint lowest test error: ${winners.map(m=>m.model).join(' / ')}`
      : `Lowest test error: ${best.model}`;
    let verdict=`${best.model} has a test error (RMSE) of ₹${fmt(best.rmse)} and an average error of ₹${fmt(best.mae)} per share.`;
    if (winners.length===1 && best.model!=='Black-Scholes' && baseline && baseline.rmse>0) {
      verdict+=` Its RMSE is ${((1-best.rmse/baseline.rmse)*100).toFixed(1)}% lower than Black-Scholes in this run.`;
    } else if (winners.length===1 && best.model==='Black-Scholes') {
      verdict+=' Black-Scholes outperformed all three neural networks in this run.';
    }
    el('option-verdict').textContent=verdict;
    table('option-metrics',['Rank','Model','Test error (RMSE) ₹ ↓','Average error (MAE) ₹ ↓','R² ↑'],ranked.map(m=>[ranked.findIndex(other=>Math.abs(other.rmse-m.rmse)<1e-9)+1,m.model,fmt(m.rmse),fmt(m.mae),fmt(m.r2)]),winners.map((_,i)=>i));
    errorChart(ranked); priceChart(data.contracts,target);
    el('option-target-label').textContent=observed?'Closer to the black reference line means a better match to uploaded prices.':'Closer to the black reference line means a better match to simulated prices.';
    const rows=[...data.contracts].sort((a,b)=>b.date.localeCompare(a.date)).slice(0,50);
    table('option-contracts',['Date','Expiry / days','Spot ₹','Strike ₹',`${target} ₹`,...names.map(n=>`${n} ₹`),...(observed?[]:['MC SE ₹'])],
      rows.map(r=>[r.date,r.expiry || `${r.days} days`,fmt(r.spot),fmt(r.strike),fmt(r.target),...names.map(n=>fmt(r.predictions[n])),...(observed?[]:[fmt(r.mc_se)])]));
    el('option-result').hidden=false;
  }
  window.runOptionComparison = async data => {
    latest=null; el('option-result').hidden=true; el('option-status').className='';
    el('option-status').textContent=`Fetching ${data.get('symbol')} history and training option models…`;
    try {
      const response=await fetch('/api/stock-options',{method:'POST',body:data});
      const result=await response.json();
      if (!response.ok) throw new Error(result.error || 'Option comparison failed.');
      render(result); el('option-status').textContent=`${result.symbol} option comparison ready.`;
    } catch(error) { el('option-status').className='error'; el('option-status').textContent=error.message; }
  };
  el('download-options').addEventListener('click',()=> {
    if (!latest) return;
    const headers=['symbol','mode','date','expiry','days','spot','strike','volatility','rate','dividend_yield','target',...names,'mc_standard_error'];
    const rows=latest.contracts.map(r=>[latest.symbol,latest.mode,r.date,r.expiry || '',r.days,r.spot,r.strike,r.volatility,latest.rate,latest.dividend,r.target,...names.map(n=>r.predictions[n]),r.mc_se]);
    const csv=[headers,...rows].map(row=>row.map(v=>`"${String(v).replaceAll('"','""')}"`).join(',')).join('\r\n');
    const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8;'}));
    const link=document.createElement('a'); link.href=url; link.download=`${latest.symbol}-${latest.mode}-option-results.csv`; link.click();
    setTimeout(()=>URL.revokeObjectURL(url),1000);
  });
})();
