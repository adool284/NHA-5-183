"""Offline, hash-bound curation. New images stay quarantined."""
import hashlib,json
from pathlib import Path
from PIL import Image,ImageOps
from collect import ROOT,save

def read(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8'))

def checked(raw):
    p=(ROOT/raw['image']).resolve()
    if not p.is_relative_to(ROOT) or not p.is_file(): raise ValueError('External/missing image')
    if hashlib.sha256(p.read_bytes()).hexdigest()!=raw['sha256']: raise ValueError('Checksum mismatch: '+str(p))
    with Image.open(p) as im:
        im=ImageOps.exif_transpose(im).convert('RGB');size=im.size
        pixel=hashlib.sha256(str(size).encode()+im.tobytes()).hexdigest()
        v=list(im.convert('L').resize((9,8)).getdata())
        dh=sum((v[y*9+x]>v[y*9+x+1])<<(y*8+x) for y in range(8) for x in range(8))
    return {**raw,'image':'../'+raw['image'],'pixel_sha256':pixel,'dhash':f'{dh:016x}','width':size[0],'height':size[1]}

def validate_splits(groups):
    seen={}
    for split,rows in groups.items():
        for r in rows:
            for f in ('sha256','pixel_sha256','original_sha256','capture_group'):
                if not r.get(f):continue
                key=(f,r[f])
                if key in seen and seen[key]!=split:raise ValueError('Cross-split overlap: '+f)
                seen[key]=split

def main():
    active={r['id'] for r in read('catalog.json')};decisions=read('review/decisions.json')
    groups={'reference':[],'validation':[],'test':[]};queue=[];rejected=[];duplicates=[];seen={}
    for raw in read('download_results.json')+read('independent_results.json'):
        if 'error' in raw:continue
        r=checked(raw)
        if r.get('artifact_id') is not None and r['artifact_id'] not in active:raise ValueError('Unknown identity')
        d=decisions.get(r['image_id'],{})
        if d.get('sha256')!=r['sha256']:d={'status':'quarantine','reason':'Unreviewed or changed image'}
        r.update(review_status=d['status'],review_reason=d.get('reason'),reviewer=d.get('reviewer'))
        if d['status']!='approved':
            (rejected if d['status']=='rejected' else queue).append(r);continue
        if not r.get('license') or not r.get('capture_group'):raise ValueError('Missing provenance')
        split=d['split'];r['split']=split
        r['evaluation_use']='previously_exposed_pilot' if split!='reference' else 'reference_only'
        key=r['pixel_sha256']
        if key in seen:
            prior=seen[key]
            if (prior['split'],prior['artifact_id'])!=(split,r['artifact_id']):raise ValueError('Duplicate crosses labels/splits')
            duplicates.append({'image_id':r['image_id'],'duplicate_of':prior['image_id']});continue
        seen[key]=r;groups[split].append(r)
    source=list(groups['reference'])
    groups['reference'].extend(checked(r) for r in read('catalog_references.json'))
    validate_splits(groups)
    rows=[r for rr in groups.values() for r in rr];near=[]
    for i,a in enumerate(rows):
        for b in rows[i+1:]:
            dist=(int(a['dhash'],16)^int(b['dhash'],16)).bit_count()
            if dist<=3:near.append({'a':a['image_id'],'b':b['image_id'],'distance':dist,'cross_split':a['split']!=b['split'],'note':'Candidate only; not proof of duplication'})
    if any(r['cross_split'] for r in near):raise ValueError('Cross-split similarity requires review')
    for name,data in [('source_gallery',source),('combined_reference',groups['reference']),('validation',groups['validation']),('test',groups['test']),('review_queue',queue),('rejected',rejected),('duplicates',duplicates),('near_duplicates',near)]:save(ROOT/f'manifests/{name}.json',data)
    summary={'active_artifacts':len(active),'source_gallery_images':len(source),'reference_images':len(groups['reference']),'reference_artifacts':len({r['artifact_id'] for r in groups['reference']}),'validation_images':len(groups['validation']),'test_images':len(groups['test']),'quarantined_images':len(queue),'rejected_images':len(rejected),'exact_duplicates_removed':len(duplicates),'near_duplicate_candidates':len(near),'source_artifacts_without_new_images':sorted(active-{r['artifact_id'] for r in source}),'final_test_ready':False,'evaluation_status':'Previously exposed pilot data; fresh independent captures needed for final acceptance.'}
    save(ROOT/'manifests/summary.json',summary);print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
