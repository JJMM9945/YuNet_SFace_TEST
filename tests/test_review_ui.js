// Dependency-free JS smoke tests. Run: node tests/test_review_ui.js
const fs=require('fs'),vm=require('vm'),assert=require('node:assert/strict');
class Element {
 constructor(tag='div'){this.tag=tag;this.children=[];this.value='';this.hidden=false;this.classList={toggle(){}};}
 append(...nodes){for(const n of nodes){if(typeof n==='object')n.parentElement=this;this.children.push(n);}}
 replaceChildren(...nodes){this.children=[];this.append(...nodes);}
 setCustomValidity(s){this.invalid=s;}
 reportValidity(){return !this.invalid;}
 remove(){} click(){}
}
const data={context:'source-context',faces:[
 {face_id:'a',photo_id:'p1',group_id:'person001',label:'인물001',crop_path:'crops/a.jpg',relative_path:'한글, "사진".jpg',status:'grouped'},
 {face_id:'b',photo_id:'p2',group_id:'person001',label:'인물001',crop_path:'crops/b.jpg',relative_path:'b.jpg',status:'grouped'},
 {face_id:'c',photo_id:'p3',group_id:'person002',label:'인물002',crop_path:'crops/c.jpg',relative_path:'c.jpg',status:'grouped'}]};
const ids={};for(const id of ['data','message','filter','groups','progress','export','import','preview','show-preview'])ids[id]=new Element();
ids.data.textContent=JSON.stringify(data);ids.filter.value='all';ids.preview.parentElement=new Element('details');
const storage=new Map();let exported;
const sandbox={document:{getElementById:id=>ids[id],createElement:tag=>new Element(tag),body:new Element('body')},
 localStorage:{setItem:(k,v)=>storage.set(k,v),getItem:k=>storage.get(k)||null},Blob,
 URL:{createObjectURL:b=>{exported=b;return 'blob:test';},revokeObjectURL(){}},setTimeout:fn=>fn()};
vm.createContext(sandbox);
const template=fs.readFileSync('face_lab/review_ui.html','utf8');
const script=template.split('<script>')[1].split('</script>')[0];vm.runInContext(script,sandbox);
const run=js=>vm.runInContext(js,sandbox);
async function main(){
 assert.equal(run('[...states.values()].filter(s=>s.manual_person_id).length'),0,'automatic groups must not be treated as truth');
 run("const firstSection=panels[0].box;firstSection.children[1].value='A01';firstSection.children[2].onclick();");
 assert.equal(run("states.get('a').manual_person_id"),'A01');assert.equal(run("states.get('b').manual_person_id"),'A01');
 run("cards.get('b').input.value='B02';cards.get('b').input.oninput();");
 assert.equal(run("states.get('b').manual_person_id"),'B02');
 ids.export.onclick();const bytes=new Uint8Array(await exported.arrayBuffer());assert.deepEqual([...bytes.slice(0,3)],[239,187,191]);
 const csv=await exported.text();sandbox.csv=csv;
 assert.equal(run('parseCSV(csv)[0].relative_path'),'한글, "사진".jpg');
 sandbox.partial='face_id,manual_person_id,include,_context_sha256\r\na,C03,1,source-context\r\n';
 ids.import.files=[{text:async()=>sandbox.partial}];await ids.import.onchange({target:ids.import});
 assert.equal(run("states.get('a').manual_person_id"),'C03');assert.equal(run("states.get('b').manual_person_id"),'');
 run("cards.get('a').input.value='D04';cards.get('a').input.oninput();");assert.equal(run("states.get('a').manual_person_id"),'D04','editing after import must update current state');
 sandbox.partial=sandbox.partial.replace('source-context','wrong-context');
 await ids.import.onchange({target:ids.import});assert.equal(run("states.get('a').manual_person_id"),'D04','failed import must not mutate annotations');
 // Import still works when file:// localStorage is unavailable.
 sandbox.localStorage.setItem=()=>{throw Error('disabled');};sandbox.partial='face_id,manual_person_id,include,_context_sha256\r\nc,E05,1,source-context\r\n';
 await ids.import.onchange({target:ids.import});assert.equal(run("states.get('c').manual_person_id"),'E05');
 run("document.getElementById('show-preview').onclick()");assert.ok(ids.preview.children.length);
 console.log('Review UI smoke checks passed: batch fill, split, CSV round-trip, provenance, import/edit, storage fallback, preview.');
}
main().catch(e=>{console.error(e);process.exitCode=1;});
