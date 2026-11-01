"""Headless Edge offline UI smoke test; output and screenshots under the suite."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
from train import ROOT,save_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--suite',default='scheduler60');args=parser.parse_args()
    folder=ROOT/'runs'/args.suite;errors=[];checks=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path=r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',headless=True)
        page=browser.new_page(viewport={'width':1440,'height':1000},device_scale_factor=1)
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto((folder/'dashboard.html').as_uri());page.wait_for_selector('#app:not([hidden])')
        assert page.locator('#summary tbody tr').count()==5
        for pde in ['heat2d','ns2d']:
            page.select_option('#pde',pde)
            fields=page.locator('#field option').evaluate_all('(nodes)=>nodes.map(n=>n.value)')
            for field in fields:
                page.select_option('#field',field)
                for value in ['0','500','1000']:
                    page.locator('#time').evaluate('(e,value)=>{e.value=value;e.dispatchEvent(new Event("input"))}',value)
                    assert page.locator('#fields canvas').count()==12
                checks.append(dict(pde=pde,field=field,passed=True))
            page.screenshot(path=str(folder/f'{pde}_dashboard.png'),full_page=True)
            page.click('#play');page.wait_for_timeout(400);assert page.locator('#time').input_value()!='0';page.click('#play')
        page.set_viewport_size({'width':430,'height':900});page.wait_for_timeout(200)
        page.screenshot(path=str(folder/'mobile_dashboard.png'),full_page=True)
        assert page.evaluate('document.documentElement.scrollWidth')<=430
        browser.close()
    save_json(folder/'browser_validation.json',dict(passed=not errors,checks=checks,errors=errors))
    assert not errors,errors
    print('Offline Edge UI check passed: both PDEs, all fields, slider, playback and mobile layout.')


if __name__=='__main__':main()
