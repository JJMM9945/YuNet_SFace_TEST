// Dependency-free simulation of the local review screen's selection and export.
const fs=require('fs'),vm=require('vm'),assert=require('node:assert/strict');
const registry={};
class Element {
 constructor(tag='div'){this.tagName=tag.toUpperCase();this.children=[];this.value='';this.hidden=false;this.attributes={};this.files=[];}
 set id(value){this._id=value;registry[value]=this;} get id(){return this._id;}
 append(...items){for(const item of items){if(typeof item==='object')item.parent=this;this.children.push(item);}}
 setAttribute(k,v){this.attributes[k]=v;} click(){} remove(){}
}
for(const id of ['review-data','queries','message','filter','progress','download','import'])registry[id]=new Element();
registry.filter.value='all';
const provenance={schema:1,source_context_sha256:'ctx',source_faces_sha256:'faces',experiment_faces_sha256:'exp',
 features_sha256:'features',comparison_sha256:'compare',representative_settings:{representatives:3}};
const f=(id,photo)=>({face_id:id,photo_id:photo,relative_path:id+'.jpg',source_group:'person001',similarity:.5,same_photo:false});
const data={provenance,queries:[
 {kind:'singleton',face_id:'a',photo_id:'p1',relative_path:'a.jpg',group_id:'person001',quality_reason:'',review_reason:'score_below_threshold',candidates:[{group_id:'person002',member_count:1,score:.5,support:1,flagged:false,representatives:[f('b','p2')]}]},
 {kind:'singleton',face_id:'b',photo_id:'p2',relative_path:'b.jpg',group_id:'person002',quality_reason:'',review_reason:'score_below_threshold',candidates:[{group_id:'person001',member_count:1,score:.5,support:1,flagged:false,representatives:[f('a','p1')]}]}
]};
registry['review-data'].textContent=JSON.stringify(data);
const memory=new Map();let exported;
const sandbox={document:{getElementById:id=>registry[id],createElement:tag=>new Element(tag),body:new Element('body')},
 localStorage:{setItem:(k,v)=>memory.set(k,v),getItem:k=>memory.get(k)||null},Blob,
 URL:{createObjectURL:b=>{exported=b;return 'blob:test';},revokeObjectURL(){}},setTimeout:fn=>fn()};
vm.createContext(sandbox);vm.runInContext(fs.readFileSync('tests/experiments/face_lab/singleton_review_ui.js','utf8'),sandbox);
function descendants(el,tag){let out=[];for(const c of el.children){if(c&&typeof c==='object'){if(c.tagName===tag)out.push(c);out=out.concat(descendants(c,tag));}}return out;}
async function main(){
 assert.equal(registry.queries.children.length,2);
 const [a,b]=registry.queries.children.map(card=>descendants(card,'SELECT')[0]);
 a.value='same';a.onchange();assert.equal(b.value,'same','reverse pair must mirror the decision');
 b.value='different';b.onchange();assert.equal(a.value,'different');
 assert.match(registry.progress.textContent,/1개/);
 registry.download.onclick();const saved=JSON.parse(await exported.text());
 assert.equal(saved.decisions.length,1);assert.equal(saved.decisions[0].decision,'different');
 registry.import.files=[{text:async()=>JSON.stringify({...saved,provenance:{...provenance,features_sha256:'wrong'}})}];
 await registry.import.onchange({target:registry.import});assert.equal(a.value,'different','invalid import must not mutate decisions');
 assert.match(registry.message.textContent,/다른 분석/);
 console.log('Singleton review DOM interaction checks passed.');
}
main().catch(e=>{console.error(e);process.exitCode=1;});
