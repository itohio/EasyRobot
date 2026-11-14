package main

// page is the client: live image, measured rate and size, controls.
const page = `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>client</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body{font:14px system-ui,sans-serif;margin:0;background:#111;color:#ddd;display:flex;flex-wrap:wrap}
#view{flex:1 1 640px;padding:8px}#view img{width:100%;height:auto;background:#000;display:block}
#side{flex:0 0 330px;padding:8px 12px;box-sizing:border-box}
fieldset{border:1px solid #444;margin:0 0 10px;padding:6px 10px}legend{color:#9cf}
label{display:block;margin:4px 0}input[type=range]{width:100%}input[type=number]{width:110px}
button{padding:4px 10px;margin:2px 0}.mono{font-family:ui-monospace,monospace;font-size:12px;white-space:pre-wrap}
#err{color:#f88}
</style></head><body>
<div id="view"><img id="live" src="/stream.mjpg" alt="live image"></div>
<div id="side">
<h3 style="margin:4px 0 8px">client</h3>
<div class="mono" id="info">connecting…</div>
<div class="mono" id="err"></div>
<fieldset><legend>exposure</legend>
<label><input type="checkbox" id="ae"> auto</label>
<input type="range" id="expR" min="0" max="1000" step="1">
<label><input type="number" id="expN" min="100" max="3000000" step="100"> µs <button id="expSet">set</button></label>
</fieldset>
<fieldset><legend>gain (ISO ≈ 100 × gain)</legend>
<label><input type="checkbox" id="ag"> auto</label>
<input type="range" id="gainR" min="1" max="15.5" step="0.1">
<label><input type="number" id="gainN" min="1" max="15.5" step="0.1"> × <button id="gainSet">set</button></label>
</fieldset>
<fieldset><legend>image</legend>
<label>size <select id="size">
<option value="0">2048×1536</option><option value="1">1600×1200</option><option value="2">1024×768</option>
<option value="3">800×600</option><option value="4">640×480</option></select></label>
<label>quality <input type="number" id="q" min="25" max="63" step="1"> (25 best … 63) <button id="qSet">set</button></label>
<label><input type="checkbox" id="hm"> mirror left-right</label>
<label><input type="checkbox" id="vf"> flip top-bottom</label>
</fieldset>
<fieldset><legend>stream</legend>
<button id="start">start</button> <button id="stop">stop</button>
<button id="snap">snapshot</button>
<div class="mono" id="snapOut"></div>
</fieldset>
</div>
<script>
const $=id=>document.getElementById(id);
const EMIN=100,EMAX=3000000;
const r2e=v=>Math.round(EMIN*Math.pow(EMAX/EMIN,v/1000));
const e2r=e=>Math.round(1000*Math.log(Math.max(e,EMIN)/EMIN)/Math.log(EMAX/EMIN));
let editing=false;
async function cmd(c){editing=false;const r=await fetch('/cmd',{method:'POST',body:new URLSearchParams({c})});if(!r.ok)$('err').textContent=await r.text();}
for(const el of document.querySelectorAll('input,select'))el.addEventListener('focus',()=>editing=true);
$('expR').oninput=()=>{editing=true;$('expN').value=r2e(+$('expR').value)};
$('expR').onchange=()=>cmd('E '+r2e(+$('expR').value));
$('expSet').onclick=()=>cmd('E '+Math.round(+$('expN').value));
$('ae').onchange=()=>cmd($('ae').checked?'E a':'E '+Math.round(+$('expN').value||10000));
$('gainR').oninput=()=>{editing=true;$('gainN').value=$('gainR').value};
$('gainR').onchange=()=>cmd('I '+$('gainR').value);
$('gainSet').onclick=()=>cmd('I '+$('gainN').value);
$('ag').onchange=()=>cmd($('ag').checked?'I a':'I '+($('gainN').value||1));
$('size').onchange=()=>cmd('Z '+$('size').value);
$('qSet').onclick=()=>cmd('Q '+$('q').value);
$('hm').onchange=$('vf').onchange=()=>cmd('M '+(+$('hm').checked)+' '+(+$('vf').checked));
$('start').onclick=()=>{cmd('S');$('live').src='/stream.mjpg?'+Date.now()};
$('stop').onclick=()=>cmd('X');
$('snap').onclick=async()=>{const r=await fetch('/snapshot',{method:'POST'});$('snapOut').textContent=r.ok?(await r.json()).path:await r.text();};
async function poll(){
 try{const s=await(await fetch('/status')).json();const b=s.board||{};
  const ms=b.exp_us?(b.exp_us/1000).toFixed(1)+' ms':'?';
  $('info').textContent='fps '+s.fps.toFixed(2)+' (board '+(b.fps||'?')+')\n'+s.width+'×'+s.height+'  '+(s.bytes/1024).toFixed(0)+' KB  #'+s.frames+'  age '+s.age_ms+' ms\n'+
   'exposure '+ms+(b.ae==='1'?' (auto)':'')+'  gain '+(b.gain||'?')+(b.ag==='1'?' (auto)':'')+'\nquality '+(b.q||'?')+'  stream '+(b.stream==='1'?'on':'off');
  $('err').textContent=s.error||'';
  if(!editing&&b.exp_us){$('ae').checked=b.ae==='1';$('expN').value=b.exp_us;$('expR').value=e2r(+b.exp_us);
   $('ag').checked=b.ag==='1';$('gainN').value=b.gain;$('gainR').value=b.gain;$('size').value=b.size;$('q').value=b.q;$('hm').checked=b.hm==='1';$('vf').checked=b.vf==='1';}
 }catch(e){$('info').textContent='client not reachable';}
 setTimeout(poll,500);}
poll();
</script></body></html>
`
