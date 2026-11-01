"""Draw the actual assumed geometry and recorded sensor positions, without inventing an indenter."""
import json
from pathlib import Path


def main():
    root = Path(__file__).parent
    data = json.loads((root/'runs/baseline/results.json').read_text(encoding='utf-8'))
    case = next(p for p in data['problems'] if p['id'] == 'contact2d')
    dots = ''.join(f'<circle cx="{150+640*x:.3f}" cy="{520-320*y:.3f}" r="4" fill="white" stroke="#286eb0" stroke-width="1.5"/>' for x, y in case['sensors'])
    arrows = ''.join(f'<path d="M {x} 138 V 194" stroke="#d05d3f" stroke-width="3" marker-end="url(#arrow)"/>' for x in range(302, 651, 35))
    supports = ''.join(f'<path d="M {x} 528 l -14 17" stroke="#7891a4" stroke-width="1.5"/>' for x in range(155, 795, 20))
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1160 690" role="img" aria-labelledby="title desc">
<title id="title">二维接触载荷反演：几何、约束与观测位置</title><desc id="desc">40毫米宽、20毫米高、10毫米厚的矩形弹性块，顶部8到32毫米区间承受未知法向压力，底部固定，左右自由。48个内部位移观测。未指定压头形状。</desc>
<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0 L10 5 L0 10Z" fill="#d05d3f"/></marker><marker id="dim" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M10 0 L0 5 L10 10" fill="none" stroke="#536f84" stroke-width="1.5"/></marker></defs>
<rect width="1160" height="690" fill="#fff"/>
<g font-family="Microsoft YaHei, system-ui, sans-serif" fill="#183044">
<text x="45" y="40" font-size="24" font-weight="700">矩形弹性块的法向接触载荷反演</text>
<text x="45" y="72" font-size="16" fill="#637b8c">外部接触体未建模：没有指定球头、圆柱、平头或压头半径。</text>
<path d="M278 113 V91 M662 113 V91 M278 100 H662" fill="none" stroke="#536f84" marker-start="url(#dim)" marker-end="url(#dim)"/>
<text x="470" y="89" text-anchor="middle" font-size="16">已知接触区：X = 8–32 mm（宽 24 mm）</text>
<rect x="150" y="200" width="640" height="320" rx="2" fill="#edf5fc" stroke="#286eb0" stroke-width="2.5"/>
<path d="M278 200 H662" stroke="#d05d3f" stroke-width="7"/>
ARROWS
<text x="470" y="126" text-anchor="middle" font-size="16" fill="#b14930">未知法向压力 p(X) ≥ 0 · 无摩擦</text>
<text x="203" y="235" font-size="13" fill="#637b8c">顶部区外：p = 0</text><text x="682" y="235" font-size="13" fill="#637b8c">p = 0</text>
<g id="sensorPositions">DOTS</g>
<path d="M150 526 H790" stroke="#536f84" stroke-width="4"/>SUPPORTS
<text x="470" y="570" text-anchor="middle" font-size="17">底部固定：u = v = 0</text>
<path d="M150 582 V619 M790 582 V619 M150 608 H790" fill="none" stroke="#536f84"/>
<path d="M155 608 H785" stroke="#536f84" marker-start="url(#dim)" marker-end="url(#dim)"/>
<text x="470" y="639" text-anchor="middle" font-size="18">宽 W = 40 mm</text>
<path d="M131 200 H86 M131 520 H86 M100 205 V515" fill="none" stroke="#536f84" marker-start="url(#dim)" marker-end="url(#dim)"/>
<text transform="translate(73 360) rotate(-90)" text-anchor="middle" font-size="18">高 H = 20 mm</text>
<text transform="translate(137 360) rotate(-90)" text-anchor="middle" font-size="14" fill="#637b8c">侧面无牵引</text>
<text transform="translate(815 360) rotate(90)" text-anchor="middle" font-size="14" fill="#637b8c">侧面无牵引</text>
<g transform="translate(860 210)" font-size="17"><text y="0" font-weight="700">材料与二维假设</text><text y="32">线弹性 · 平面应力</text><text y="62">E = 1 MPa，ν = 0.3</text><text y="92">出平面厚度 b = 10 mm</text><circle cy="139" cx="6" r="4" fill="white" stroke="#286eb0" stroke-width="1.5"/><text y="145" x="20">48 个内部位移观测点</text><text y="194" font-weight="700">根据位移估计</text><text y="226">压力分布 p(X)</text><text y="258">合力 F = b ∫p(X)dX</text><text y="302" font-size="14" fill="#637b8c">箭头仅表示受力方向。</text><text y="328" font-size="14" fill="#637b8c">不表示某一种压头的轮廓。</text></g>
<text x="45" y="678" font-size="14" fill="#637b8c">合成验证；已知接触区。未求解未知间隙、接触区搜索、摩擦或压头形状反演。</text>
</g></svg>'''.replace('ARROWS', arrows).replace('DOTS', dots).replace('SUPPORTS', supports)
    assets = root/'assets'
    assets.mkdir(exist_ok=True)
    (assets/'contact_geometry.svg').write_text(svg, encoding='utf-8')


if __name__ == '__main__':
    main()
