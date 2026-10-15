"""Render recorded MuJoCo states and build a static interactive experiment report."""
import argparse,json,math
from pathlib import Path
import mujoco
import numpy as np

def render_videos(report_dir):
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw
    report_dir=Path(report_dir)
    replays=json.loads((report_dir/'replays.json').read_text())
    xml=Path(__file__).with_name('humanoid.xml').read_text()
    # Rendering-only materials and lighting; dynamics and joint order are unchanged.
    xml=xml.replace('<worldbody>','''<asset>
      <texture name="tiles" type="2d" builtin="checker" rgb1=".13 .18 .24" rgb2=".19 .25 .32" width="128" height="128"/>
      <material name="floor_tiles" texture="tiles" texrepeat="100 10" texuniform="true" reflectance=".05"/>
      <texture type="skybox" builtin="gradient" rgb1=".05 .08 .13" rgb2=".2 .28 .36" width="128" height="128"/>
    </asset><worldbody>''').replace('name="floor" type="plane"','name="floor" material="floor_tiles" type="plane"')
    m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m)
    m.vis.headlight.ambient[:]=[.5,.5,.5];m.vis.headlight.diffuse[:]=[.8,.8,.8]
    with mujoco.Renderer(m,height=360,width=640) as renderer:
        cam=mujoco.MjvCamera();cam.distance=2.7;cam.azimuth=105;cam.elevation=-12
        for name,replay in replays.items():
            with imageio.get_writer(report_dir/(name+'.mp4'),fps=100/3,codec='libx264',quality=7,macro_block_size=1,
                                    ffmpeg_params=['-movflags','+faststart']) as writer:
                for i,frame in enumerate(replay['frames']):
                    d.qpos[:]=frame['qpos'];mujoco.mj_forward(m,d)
                    cam.lookat[:]=[d.qpos[0]+.18,0,.70]
                    renderer.update_scene(d,camera=cam)
                    picture=Image.fromarray(renderer.render());draw=ImageDraw.Draw(picture)
                    draw.rectangle((0,0,640,35),fill=(12,21,33))
                    draw.text((14,10),f"{replay['label'].replace('_',' ')}   |   t={frame['t']:.2f}s   x={frame['qpos'][0]:.2f}m",fill=(226,238,247))
                    draw.text((14,340),'RECORDED MUJOCO PHYSICS / PLANAR CONSTRAINT / NO REAL ROBOT',fill=(200,216,230))
                    writer.append_data(np.asarray(picture))
                    if name=='learned_nominal' and i==100:picture.save(report_dir/'preview.png')
            print('Rendered',name,flush=True)

def build_report(report_dir):
    report_dir=Path(report_dir)
    data=json.loads((report_dir/'evaluation.json').read_text(encoding='utf-8'))
    replays=json.loads((report_dir/'replays.json').read_text(encoding='utf-8'))
    data['replays']={k:{'label':v['label'],'metrics':v['metrics'],
        'frames':[{key:frame[key] for key in ('t','velocity','height','contacts')} for frame in v['frames']]}
        for k,v in replays.items()}
    template=Path(__file__).with_name('template.html').read_text(encoding='utf-8')
    training=data['training'];horizon=data['duration_s'];count=data['summary']['learned_nominal']['trials']
    template=template.replace('12 seconds each',f'{horizon:g} seconds each')
    template=template.replace('50 generations',f"{len(training['history'])} generations")
    template=template.replace('20 runs per condition',f'{count} runs per condition')
    template=template.replace('Sample 48 candidate sets',f"Sample {training['population']} candidate sets")
    template=template.replace('up to six seconds',f"up to {training['seconds']:g} seconds")
    template=template.replace('for twelve seconds',f'for {horizon:g} seconds')
    (report_dir/'report.html').write_text(template.replace('__DATA__',json.dumps(data,ensure_ascii=True).replace('<','\\u003c')),encoding='utf-8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='reports');p.add_argument('--html-only',action='store_true');a=p.parse_args()
    if not a.html_only:render_videos(a.out)
    build_report(a.out)
if __name__=='__main__':main()
