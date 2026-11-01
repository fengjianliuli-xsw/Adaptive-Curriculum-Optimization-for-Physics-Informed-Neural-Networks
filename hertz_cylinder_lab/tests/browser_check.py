import argparse
from pathlib import Path
from playwright.sync_api import sync_playwright


def main():
    parser=argparse.ArgumentParser();parser.add_argument('dashboard',type=Path);args=parser.parse_args()
    path=args.dashboard.resolve();errors=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(channel='msedge',headless=True)
        page=browser.new_page(viewport={'width':1440,'height':1000})
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(path.as_uri(),wait_until='load');page.wait_for_function('window.hertzLab')
        assert page.evaluate('hertzLab.getState().fair')
        count=page.evaluate('hertzLab.data.runs.length')
        assert page.locator('#table tbody tr').count()==count
        page.locator('#reset').click();assert page.evaluate('hertzLab.getState().progress')==0
        page.locator('#play').click();page.wait_for_function('hertzLab.getState().progress>.01')
        page.locator('#play').click();assert not page.evaluate('hertzLab.getState().playing')
        page.locator('#axis').select_option('step');page.locator('#all').click()
        for box in page.locator('#legend input').all():box.uncheck()
        page.locator('#focus').click()
        page.locator('#progress').evaluate("e=>{e.value=1000;e.dispatchEvent(new Event('input'));}")
        page.screenshot(path=str(path.parent/'dashboard_preview.png'),full_page=True)
        page.set_viewport_size({'width':390,'height':844});page.evaluate('hertzLab.render()')
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
        assert not errors,errors
        browser.close()
    print(f'Browser checks passed: {count} methods, playback, toggles, axes, mobile; no script errors.')


if __name__=='__main__':main()
