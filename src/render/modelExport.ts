import {
  BufferAttribute,
  BufferGeometry,
  InstancedBufferAttribute,
  InstancedMesh,
  Matrix3,
  Matrix4,
  Mesh,
  MeshStandardMaterial,
  Quaternion,
  Scene,
  Vector3,
} from 'three';
import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';
import { createDisplayColor } from './displayColor';

const DEG_TO_RAD = Math.PI / 180;
const OBJ_ASYNC_VERTEX_BUDGET = 15_360;

export interface ExportInstanceData {
  matrices: Float32Array;
  birthRanks: Float32Array;
  count: number;
  seedCount: number;
  gradientCount: number;
  spherePositions: Float32Array;
  sphereNormals: Float32Array;
  sphereScale: number;
  rotationDegrees: number;
  innerColor: string;
  outerColor: string;
  brightness: number;
  contrast: number;
  materialRoughness: number;
}

/**
 * Produces one RGB triplet per particle. Seed particles stay at the inner color;
 * the newest displayed attached particle reaches the outer color exactly.
 */
export function createAgeGradientColors(data: ExportInstanceData): Float32Array {
  const colors = new Float32Array(data.count * 3);
  const inner = createDisplayColor(data.innerColor);
  const outer = createDisplayColor(data.outerColor);
  const attachedCount = Math.max(1, data.gradientCount - Math.max(0, data.seedCount));

  for (let i = 0; i < data.count; i++) {
    const rank = data.birthRanks[i] ?? i;
    const age = rank < data.seedCount
      ? 0
      : clamp01((rank - data.seedCount + 1) / attachedCount);
    const offset = i * 3;
    colors[offset] = gradeColor(inner.r + (outer.r - inner.r) * age, data.brightness, data.contrast);
    colors[offset + 1] = gradeColor(inner.g + (outer.g - inner.g) * age, data.brightness, data.contrast);
    colors[offset + 2] = gradeColor(inner.b + (outer.b - inner.b) * age, data.brightness, data.contrast);
  }

  return colors;
}

/** Creates a compact GLB using EXT_mesh_gpu_instancing. */
export async function createGlbBlob(data: ExportInstanceData): Promise<Blob> {
  assertExportData(data);
  const scene = createInstancedExportScene(data);
  const exporter = new GLTFExporter();

  try {
    const result = await exporter.parseAsync(scene, {
      binary: true,
      forceIndices: false,
      onlyVisible: true,
    });
    if (!(result instanceof ArrayBuffer)) {
      throw new Error('GLB export did not produce a binary buffer.');
    }
    return new Blob([result], { type: 'model/gltf-binary' });
  } finally {
    disposeScene(scene);
  }
}

/**
 * Expands every sphere triangle for OBJ. Work is yielded in bounded batches so
 * large exports do not monopolize the browser event loop.
 */
export async function createObjBlob(data: ExportInstanceData): Promise<Blob> {
  assertExportData(data);
  const chunks: string[] = [
    '# 260716_DLAFractals\n',
    'o DLAFractals\n',
    's 1\n',
  ];
  const colors = createAgeGradientColors(data);
  const vertexCount = data.spherePositions.length / 3;
  const instancesPerBatch = Math.max(1, Math.floor(OBJ_ASYNC_VERTEX_BUDGET / vertexCount));
  const objectMatrix = createObjectMatrix(data);
  const instanceMatrix = new Matrix4();
  const worldMatrix = new Matrix4();
  const normalMatrix = new Matrix3();
  const vertex = new Vector3();
  const normal = new Vector3();
  let batch = '';

  for (let instanceIndex = 0; instanceIndex < data.count; instanceIndex++) {
    instanceMatrix.fromArray(data.matrices, instanceIndex * 16);
    worldMatrix.multiplyMatrices(objectMatrix, instanceMatrix);
    normalMatrix.getNormalMatrix(worldMatrix);
    const colorOffset = instanceIndex * 3;

    for (let vertexIndex = 0; vertexIndex < vertexCount; vertexIndex++) {
      const sourceOffset = vertexIndex * 3;
      vertex
        .fromArray(data.spherePositions, sourceOffset)
        .applyMatrix4(worldMatrix);
      normal
        .fromArray(data.sphereNormals, sourceOffset)
        .applyMatrix3(normalMatrix)
        .normalize();

      batch += `v ${formatNumber(vertex.x)} ${formatNumber(vertex.y)} ${formatNumber(vertex.z)} ${formatNumber(colors[colorOffset])} ${formatNumber(colors[colorOffset + 1])} ${formatNumber(colors[colorOffset + 2])}\n`;
      batch += `vn ${formatNumber(normal.x)} ${formatNumber(normal.y)} ${formatNumber(normal.z)}\n`;
    }

    const firstVertex = instanceIndex * vertexCount + 1;
    for (let vertexIndex = 0; vertexIndex < vertexCount; vertexIndex += 3) {
      const a = firstVertex + vertexIndex;
      const b = a + 1;
      const c = a + 2;
      batch += `f ${a}//${a} ${b}//${b} ${c}//${c}\n`;
    }

    if ((instanceIndex + 1) % instancesPerBatch === 0) {
      chunks.push(batch);
      batch = '';
      await yieldToBrowser();
    }
  }

  if (batch.length > 0) {
    chunks.push(batch);
  }

  return new Blob(chunks, { type: 'text/plain;charset=utf-8' });
}

function createInstancedExportScene(data: ExportInstanceData): Scene {
  const scene = new Scene();
  const geometry = new BufferGeometry();
  geometry.setAttribute('position', new BufferAttribute(data.spherePositions.slice(), 3));
  geometry.setAttribute('normal', new BufferAttribute(data.sphereNormals.slice(), 3));
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();

  const material = new MeshStandardMaterial({
    color: 0xffffff,
    roughness: clamp01(data.materialRoughness),
    metalness: 0,
    flatShading: false,
    vertexColors: true,
  });
  material.name = 'DLA Age Gradient';

  const mesh = new InstancedMesh(geometry, material, data.count);
  mesh.name = '260716_DLAFractals';
  mesh.instanceMatrix.array.set(data.matrices);
  mesh.instanceMatrix.needsUpdate = true;
  mesh.instanceColor = new InstancedBufferAttribute(createAgeGradientColors(data), 3);
  mesh.instanceColor.needsUpdate = true;
  mesh.rotation.y = data.rotationDegrees * DEG_TO_RAD;
  mesh.scale.setScalar(data.sphereScale);
  mesh.count = data.count;
  mesh.updateMatrix();
  mesh.updateMatrixWorld(true);
  scene.add(mesh);
  return scene;
}

function createObjectMatrix(data: ExportInstanceData): Matrix4 {
  return new Matrix4().compose(
    new Vector3(),
    new Quaternion().setFromAxisAngle(new Vector3(0, 1, 0), data.rotationDegrees * DEG_TO_RAD),
    new Vector3(data.sphereScale, data.sphereScale, data.sphereScale),
  );
}

function assertExportData(data: ExportInstanceData): void {
  if (data.count <= 0) {
    throw new Error('No visible DLA particles are available to export.');
  }
  if (data.matrices.length < data.count * 16) {
    throw new Error('The export matrix buffer is smaller than the displayed particle count.');
  }
  if (data.birthRanks.length < data.count) {
    throw new Error('The export birth-rank buffer is smaller than the displayed particle count.');
  }
  if (data.spherePositions.length === 0 || data.spherePositions.length % 9 !== 0) {
    throw new Error('Sphere export geometry must contain complete non-indexed triangles.');
  }
  if (data.sphereNormals.length !== data.spherePositions.length) {
    throw new Error('Sphere export geometry has mismatched positions and normals.');
  }
}

function gradeColor(value: number, brightness: number, contrast: number): number {
  const contrasted = Math.min(4, Math.max(0, (value - 0.5) * Math.max(0, contrast) + 0.5));
  return clamp01(contrasted * Math.max(0, brightness));
}

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}

function formatNumber(value: number): string {
  if (!Number.isFinite(value) || Math.abs(value) < 0.000001) {
    return '0';
  }
  return Number(value.toFixed(6)).toString();
}

function yieldToBrowser(): Promise<void> {
  if (typeof requestAnimationFrame !== 'function') {
    return Promise.resolve();
  }
  return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

function disposeScene(scene: Scene): void {
  scene.traverse((object) => {
    if (!(object instanceof Mesh)) {
      return;
    }
    object.geometry.dispose();
    if (Array.isArray(object.material)) {
      object.material.forEach((material) => material.dispose());
    } else {
      object.material.dispose();
    }
    if (object instanceof InstancedMesh) {
      object.dispose();
    }
  });
}
