"""Optional: pip install playwright. Uses installed Edge, no browser download."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dashboard', type=Path)
    args = parser.parse_args()
    target = args.dashboard.resolve()
    errors, external = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='msedge', headless=True)
        page = browser.new_page(viewport={'width': 1440, 'height': 1000})
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.on('request', lambda r: external.append(r.url) if r.url.startswith(('http:', 'https:')) else None)
        root = target.parent.parent.parent
        page.goto((root/'dashboard.template.html').as_uri(), wait_until='load')
        assert page.locator('#templateNotice').is_visible()
        assert page.locator('main').is_hidden()
        page.locator('#templateResultLink').click()
        page.wait_for_function('window.pinn2d')
        assert page.locator('#table tbody tr').count() == 10
        assert page.locator('#charts canvas').count() == 5
        assert len(page.evaluate('pinn2d.getState().selected')) == 3
        page.locator('#all').click()
        assert len(page.evaluate('pinn2d.getState().selected')) == 10
        page.screenshot(path=str(target.parent/'heat2d_dashboard.png'), full_page=True)
        page.locator('#reset').click()
        assert page.evaluate('pinn2d.getState().progress') == 0
        page.locator('#play').click()
        page.wait_for_function('pinn2d.getState().progress > 0.01')
        page.locator('#play').click()
        assert not page.evaluate('pinn2d.getState().playing')
        page.locator('#axis').select_option('train_seconds')
        page.locator('#progress').evaluate("e=>{e.value=1000;e.dispatchEvent(new Event('input'));}")
        page.locator('#contactShortcut').click()
        assert page.evaluate('pinn2d.getState().problem') == 'contact2d'
        assert page.locator('#contactGeometry').is_visible()
        assert page.locator('#contactGeometry svg').count() == 1
        assert page.locator('#charts canvas').count() == 9
        assert 'N' in page.locator('#fourthValue').inner_text()
        page.locator('#field').select_option('ux_mm')
        page.locator('#view').select_option('error')
        page.locator('#view').select_option('prediction')
        page.locator('#field').select_option('uy_mm')
        page.screenshot(path=str(target.parent/'contact2d_dashboard.png'), full_page=True)
        page.locator('#all').click()
        for box in page.locator('#legend input').all():
            box.uncheck()
        assert page.evaluate('pinn2d.getState().selected.length') == 0
        page.locator('#focus').click()
        with page.expect_download() as download:
            page.locator('#downloadCsv').click()
        assert download.value.suggested_filename == 'contact2d_summary.csv'
        with page.expect_download() as download:
            page.locator('#saveFields').click()
        assert download.value.suggested_filename.endswith('.png')
        page.set_viewport_size({'width': 390, 'height': 844})
        page.evaluate('pinn2d.render()')
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
        page.screenshot(path=str(target.parent/'mobile_dashboard.png'), full_page=True)
        page.goto((root/'index.html').as_uri(), wait_until='load')
        page.get_by_role('link', name='直接查看接触力反演').click()
        page.wait_for_function("window.pinn2d && pinn2d.getState().problem === 'contact2d'")
        assert page.locator('#problem').input_value() == 'contact2d'
        page.set_viewport_size({'width': 1160, 'height': 690})
        page.goto((root/'assets/contact_geometry.svg').as_uri(), wait_until='load')
        page.screenshot(path=str(root/'assets/contact_geometry.png'))
        assert not errors, errors
        assert not external, external
        browser.close()
    print(json.dumps({'errors': errors, 'external_requests': external, 'checks': 'passed 2 PDEs / 10 methods / playback / fields / pressure / export / mobile'}))


if __name__ == '__main__':
    main()
