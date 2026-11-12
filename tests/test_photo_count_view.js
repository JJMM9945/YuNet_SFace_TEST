// Offline viewer interaction smoke checks without external browser dependencies.
const fs=require('fs'),vm=require('vm'),assert=require('assert');
class Element{
 constructor(){this.children=[];this.dataset={};this.style={};this.value='';this.hidden=false;this.attrs={};}
 append(...nodes){this.children.push(...nodes)} replaceChildren(...nodes){this.children=nodes}
 setAttribute(k,v){this.attrs[k]=v} focus(){} showModal(){this.open=true} close(){this.open=false}
}
const ids={};for(const id of ['report-data','metrics','search','filter','sort','groups','resultcount','viewer','photoTitle','photoInfo','box','photo','close','reset'])ids[id]=new Element();
ids.filter.value='all';ids.sort.value='size';
const tabs=['all','solo','pair','crowd'].map(key=>{const e=new Element();e.dataset.case=key;return e});
const photo=(id,state)=>({photo_id:id,path:'<test>.jpg',count:1,state,reason:'',faces:[{face_id:'f',box:[10,20,30,40],identity:'A',own:null,reason:''}]});
const group={id:'solo-001',title:'인물 01',identities:['A'],photos:[photo('p','assigned')]};
const unknown={id:'pair-pending',title:'확인 필요',identities:[],photos:[photo('q','held'),photo('r','quality_excluded')]};
const s={photos:3,faces:3,groups:1,singleton_groups:1,held_photos:1,quality_check_photos:1};
ids['report-data'].textContent=JSON.stringify({summary:{cases:{all:s,solo:s,pair:s,crowd:s}},cases:{all:[group,unknown],solo:[group],pair:[],crowd:[]}});
const document={getElementById:id=>ids[id],createElement:()=>new Element(),querySelectorAll:()=>tabs};
const source=fs.readFileSync('face_lab/album_report.html','utf8');
const script=source.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
vm.runInNewContext(script,{document,console});
assert.equal(ids.groups.children.length,2);
ids.filter.value='held';ids.filter.onchange();assert.equal(ids.groups.children.length,1);assert(ids.resultcount.textContent.includes('사진 1장'));
ids.reset.onclick();ids.search.value='missing';ids.search.oninput();assert.equal(ids.groups.children[0].textContent,'조건에 맞는 사진 앨범이 없습니다.');
ids.reset.onclick();tabs[1].onclick();assert.equal(ids.groups.children.length,1);assert.equal(tabs[1].attrs['aria-selected'],true);
const card=ids.groups.children[0];card.children[3].onclick();assert.equal(card.children[4].hidden,false);
card.children[1].children[0].onclick();assert.equal(ids.viewer.open,true);ids.photo.naturalWidth=100;ids.photo.naturalHeight=100;ids.photo.onload();assert.equal(ids.box.children[0].style.left,'10%');assert.equal(ids.box.children[0].style.width,'30%');
ids.close.onclick();assert.equal(ids.viewer.open,false);
tabs[2].onclick();assert(ids.resultcount.textContent.includes('0개 앨범'));
console.log('PASS: tabs, search, filter, reset, evidence details, photo dialog and face overlay');
