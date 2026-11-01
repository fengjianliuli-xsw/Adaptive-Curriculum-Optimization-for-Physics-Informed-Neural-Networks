"""Optional browser integration check: pip install playwright, uses installed Edge.

Run: python tests/browser_check.py runs/demo/dashboard.html
"""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dashboard', type=Path)
    args = parser.parse_args()
    dashboard = args.dashboard.resolve()
    errors, external_requests = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='msedge', headless=True)
        page = browser.new_page(viewport={'width': 1440, 'height': 1000}, device_scale_factor=1)
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.on('request', lambda r: external_requests.append(r.url) if r.url.startswith(('http:', 'https:')) else None)
        page.goto(dashboard.as_uri(), wait_until='load')
        page.wait_for_function('window.pinnLab && window.pinnLab.getState().fair')
        assert page.locator('#summary tbody tr').count() == 10
        assert page.locator('#charts canvas').count() == 6
        page.screenshot(path=str(dashboard.parent/'dashboard_preview.png'), full_page=True)
        page.locator('#reset').click()
        assert page.evaluate('window.pinnLab.getState().progress') == 0
        assert 'step 0' in page.locator('#predictions .meta').first.inner_text()
        page.locator('#physicalTime').select_option('4')
        page.locator('#axis').select_option('train_seconds')
        page.locator('#play').click()
        page.wait_for_function('window.pinnLab.getState().progress > 0.02')
        page.locator('#play').click()
        assert not page.evaluate('window.pinnLab.getState().playing')
        page.locator('#progress').evaluate("e => {e.value=1000;e.dispatchEvent(new Event('input'));}")
        assert page.evaluate('window.pinnLab.getState().progress') == 1
        checks = page.locator('#legend input')
        checks.first.uncheck()
        assert page.locator('#predictions .prediction').first.is_hidden()
        for box in checks.all():
            box.uncheck()
        assert page.evaluate('window.pinnLab.getState().selected.length') == 0
        for box in checks.all():
            box.check()
        with page.expect_download() as download:
            page.locator('#downloadCsv').click()
        assert download.value.suggested_filename.endswith('.csv')
        with page.expect_download() as download:
            page.locator('#savePrediction').click()
        assert download.value.suggested_filename.endswith('.png')
        page.set_viewport_size({'width': 390, 'height': 844})
        page.evaluate('window.pinnLab.render()')
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth+1')
        page.screenshot(path=str(dashboard.parent/'dashboard_mobile.png'), full_page=True)
        assert not errors, errors
        assert not external_requests, external_requests
        browser.close()
    print(json.dumps({'page_errors': errors, 'external_requests': external_requests,
                      'checks': ['initialization', 'charts', 'reset', 'time_slice', 'time_axis',
                                 'play_pause', 'scrub', 'toggle_all', 'csv_download', 'png_download',
                                 'mobile_layout']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
