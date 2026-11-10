// Pair-specific offline decisions, import validation and conflict protection.
const assert=require('node:assert/strict');
const {pairKey,allowedPairs,validateImport,exportDocument,findConflicts}=require('../tests/experiments/face_lab/singleton_review_ui.js');
const data={queries:[{face_id:'a',candidates:[{representatives:[{face_id:'b'},{face_id:'c'}]}]},
                     {face_id:'b',candidates:[{representatives:[{face_id:'a'}]}]}]};
const provenance={source_context_sha256:'ctx',source_faces_sha256:'faces',experiment_faces_sha256:'exp',
 features_sha256:'features',comparison_sha256:'compare',ground_truth_sha256:'truth',representative_settings:{representatives:3,margin:.03}};
const allowed=allowedPairs(data),state=new Map([[pairKey('a','b'),'same'],[pairKey('a','c'),'uncertain']]);
assert.equal(allowed.size,2);assert.equal(pairKey('a','b'),pairKey('b','a'));
const result=exportDocument(provenance,allowed,state);assert.equal(result.decisions.length,2);
assert.deepEqual([...validateImport(result,provenance,allowed)],[...state]);
assert.throws(()=>validateImport({...result,provenance:{...provenance,features_sha256:'other'}},provenance,allowed),/다른 분석/);
assert.throws(()=>validateImport({...result,provenance:{...provenance,ground_truth_sha256:'other'}},provenance,allowed),/다른 분석/);
assert.throws(()=>validateImport({...result,decisions:[...result.decisions,{face_a:'b',face_b:'a',decision:'different'}]},provenance,allowed),/중복/);
assert.throws(()=>validateImport({...result,decisions:[{face_a:'a',face_b:'missing',decision:'same'}]},provenance,allowed),/알 수 없거나/);
assert.throws(()=>validateImport({...result,decisions:[{face_a:'a',face_b:'b',decision:'auto'}]},provenance,allowed),/판정/);
const triangle=new Map([['a|b',['a','b']],['b|c',['b','c']],['a|c',['a','c']]]);
const conflicting=new Map([['a|b','same'],['b|c','same'],['a|c','different']]);
assert.equal(findConflicts(triangle,conflicting).length,1);
assert.equal(exportDocument(provenance,triangle,conflicting).conflicts.length,1);
console.log('Singleton review pair export/import checks passed.');
