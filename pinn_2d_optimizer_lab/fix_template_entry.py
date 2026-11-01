"""One-time migration of the existing template; kept for change traceability."""
from pathlib import Path

root = Path(__file__).parent
path = root/'dashboard.template.html'
s = path.read_text(encoding='utf-8')
needle = '<section class="panel controls">'
geometry = '''<section class="panel" id="contactGeometry" hidden><h2>这是什么接触？几何与边界条件</h2><p class="small muted">这是矩形弹性块在顶部已知区域承受未知法向压力的合成反演。没有指定外部压头的形状或半径，因此本例不属于已指定球形/圆柱形压头的赫兹接触。</p>__CONTACT_GEOMETRY_SVG__<div class="notice small">已知：弹性块尺寸、材料、接触区、底部固定、侧面自由。观测：48 个内部点的位移。估计：压力 p(X) 和合力 F。外部接触体的几何没有进入本轮 PDE。</div></section>\n'''
if '__CONTACT_GEOMETRY_SVG__' not in s:
    assert needle in s
    s = s.replace(needle, geometry+needle, 1)
old = "'use strict';\nconst D=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id);"
new = """'use strict';
(()=>{
const raw=document.getElementById('data').textContent;
let D;
try{D=JSON.parse(raw);}catch(error){document.querySelector('main').hidden=true;document.getElementById('templateNotice').hidden=false;if(!location.pathname.endsWith('/dashboard.template.html')){document.getElementById('templateResultLink').href='dashboard.html';document.getElementById('templateHomeLink').href='../../index.html';}return;}
const $=id=>document.getElementById(id);"""
if old in s:
    s = s.replace(old, new, 1)
    s = s.replace("let P=D.problems[0],selected=", "let P=D.problems.find(p=>p.id===new URL(location.href).searchParams.get('problem'))||D.problems[0],selected=", 1)
    s = s.replace("const contact=P.id==='contact2d';$('fourthTitle')", "const contact=P.id==='contact2d';$('contactGeometry').hidden=!contact;$('fourthTitle')", 1)
    s = s.replace("hover($('cutCanvas'));initialize();window.pinn2d=", "hover($('cutCanvas'));$('problem').value=P.id;initialize();window.pinn2d=", 1)
    s = s.replace('</script></body></html>', '})();\n</script></body></html>', 1)
path.write_text(s, encoding='utf-8')
