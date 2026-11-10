"""Pinned AuraFace weights and model-specific, local-only recognition inference."""
import json
from pathlib import Path
import urllib.request

import cv2
import numpy as np

from .models import sha256, verify


REVISION = 'af6d057c9b0ec4071d4c49c80e3539258798b609'
SPEC = dict(name='glintr100.onnx', size=260694151,
            sha256='a7933ea5330113b01c9b60351d8f4c33003f145d8470ac5f0e52ee2effe25c60',
            model='fal/AuraFace-v1', revision=REVISION, license='Apache-2.0')
BASE_URL = f'https://huggingface.co/fal/AuraFace-v1/resolve/{REVISION}/'
TEMPLATE = np.array([[38.2946,51.6963],[73.5318,51.5014],[56.0252,71.7366],
                     [41.5493,92.3655],[70.7299,92.2041]], dtype=np.float64)


def prepare_auraface(folder):
    """Explicit public-model downloads only. No local image or vector is transmitted."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    target=folder/SPEC['name']
    if target.exists():
        verify(target,SPEC)
    else:
        temporary=folder/(SPEC['name']+'.download')
        if temporary.exists():
            raise ValueError('Inspect the existing partial AuraFace download before retrying')
        with urllib.request.urlopen(BASE_URL+SPEC['name'],timeout=120) as response, temporary.open('xb') as file:
            total=0;last=0
            while block:=response.read(1024*1024):
                file.write(block);total+=len(block)
                if total-last>=32*1024*1024:
                    print(f'AuraFace model download: {total/1024/1024:.0f} MiB',flush=True);last=total
        verify(temporary,SPEC)
        temporary.rename(target)
    for name in ('LICENSE.md','README.md'):
        path=folder/name
        if not path.exists():
            with urllib.request.urlopen(BASE_URL+name,timeout=60) as response:
                data=response.read()
            with path.open('xb') as file:file.write(data)
    provenance=dict(SPEC,url=BASE_URL+SPEC['name'],
                    metadata_url='https://huggingface.co/api/models/fal/AuraFace-v1?blobs=true',
                    files={name:sha256(folder/name) for name in (SPEC['name'],'LICENSE.md','README.md')})
    (folder/'model_provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Verified AuraFace: {target}')
    return target


def similarity_matrix(landmarks):
    """Least-squares rotation+uniform-scale+translation, equivalent to 2D similarity alignment."""
    points=np.asarray(landmarks,dtype=np.float64)
    if points.shape!=(5,2) or not np.isfinite(points).all():
        raise ValueError('Five finite original-coordinate landmarks are required')
    if np.linalg.matrix_rank(points-points.mean(axis=0))<2:
        raise ValueError('Degenerate face landmarks')
    # x'=a*x-b*y+tx, y'=b*x+a*y+ty. No reflection or arbitrary affine shear.
    design=np.zeros((10,4),np.float64)
    design[0::2,0]=points[:,0];design[0::2,1]=-points[:,1];design[0::2,2]=1
    design[1::2,0]=points[:,1];design[1::2,1]=points[:,0];design[1::2,3]=1
    parameters,_,rank,_=np.linalg.lstsq(design,TEMPLATE.reshape(-1),rcond=None)
    if rank!=4:raise ValueError('Invalid face similarity transform')
    a,b,tx,ty=parameters
    if a*a+b*b<=1e-12:raise ValueError('Invalid face similarity scale')
    return np.array([[a,-b,tx],[b,a,ty]],np.float64)


def align_face(bgr, landmarks):
    matrix=similarity_matrix(landmarks)
    crop=cv2.warpAffine(bgr,matrix,(112,112),flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    return crop,matrix


def normalize_embedding(raw,dimension=512):
    feature=np.asarray(raw,dtype=np.float32).reshape(-1).copy()
    norm=float(np.linalg.norm(feature))
    if feature.size!=dimension or not np.isfinite(feature).all() or not np.isfinite(norm) or norm<=1e-12:
        raise ValueError('Invalid AuraFace embedding')
    return feature/norm


def input_blob(crop,mean,std):
    if crop.shape!=(112,112,3) or crop.dtype!=np.uint8:
        raise ValueError('Expected uint8 BGR aligned 112x112 image')
    return cv2.dnn.blobFromImage(crop,1.0/std,(112,112),(mean,mean,mean),swapRB=True,crop=False)


class AuraFace:
    def __init__(self,folder):
        import onnx
        import onnxruntime as ort
        path=Path(folder)/SPEC['name'];verify(path,SPEC)
        graph=onnx.load(str(path),load_external_data=False)
        first_nodes=[node.name for node in graph.graph.node[:8]]
        sub=any(name.startswith(('Sub','_minus')) for name in first_nodes)
        mul=any(name.startswith(('Mul','_mul')) for name in first_nodes)
        self.mean,self.std=(0.,1.) if sub and mul else (127.5,127.5)
        del graph
        options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
        options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
        inputs=self.session.get_inputs();outputs=self.session.get_outputs()
        if len(inputs)!=1 or len(outputs)!=1 or inputs[0].shape[1:]!=[3,112,112]:
            raise ValueError('Unexpected AuraFace input/output contract')
        self.input_name=inputs[0].name;self.output_name=outputs[0].name
        self.metadata=dict(model=SPEC,input_shape=inputs[0].shape,output_shape=outputs[0].shape,
                           input_mean=self.mean,input_std=self.std,channel='RGB through swapRB from BGR',
                           first_onnx_nodes=first_nodes,runtime='onnxruntime',runtime_version=ort.__version__,
                           onnx_version=onnx.__version__,provider='CPUExecutionProvider',threads=1,
                           alignment='112x112 ArcFace five-point least-squares similarity, black border, INTER_LINEAR',
                           embedding='512 float32, finite check, L2 normalized')
        self.feature(np.zeros((112,112,3),np.uint8))

    def feature(self,crop):
        blob=input_blob(crop,self.mean,self.std)
        raw=self.session.run([self.output_name],{self.input_name:blob})[0]
        return normalize_embedding(raw)
