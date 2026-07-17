import {
  BufferAttribute,
  BufferGeometry,
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
  seedRotationDegrees: number;
  innerColor: string;
  outerColor: string;
  gradientContrast: number;
  gradientBias: number;
  gradientBlur: number;
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
    const linearAge = rank < data.seedCount
      ? 0
      : clamp01((rank - data.seedCount + 1) / attachedCount);
    const age = shapeAgeGradient(
      linearAge,
      data.gradientContrast,
      data.gradientBias,
      data.gradientBlur,
    );
    const offset = i * 3;
    const seedParticle = rank < data.seedCount;
    const red = inner.r + (outer.r - inner.r) * age;
    const green = inner.g + (outer.g - inner.g) * age;
    const blue = inner.b + (outer.b - inner.b) * age;
    colors[offset] = seedParticle ? clamp01(red) : gradeColor(red, data.brightness, data.contrast);
    colors[offset + 1] = seedParticle ? clamp01(green) : gradeColor(green, data.brightness, data.contrast);
    colors[offset + 2] = seedParticle ? clamp01(blue) : gradeColor(blue, data.brightness, data.contrast);
  }

  return colors;
}

/**
 * Applies the DifferentialGrowth contrast/bias curve to birth age. Blur softens
 * that grading toward the underlying linear age ramp while endpoints remain exact.
 */
export function shapeAgeGradient(
  age: number,
  contrast: number,
  bias: number,
  blur: number,
): number {
  const linearAge = clamp01(age);
  if (linearAge <= 0 || linearAge >= 1) {
    return linearAge;
  }
  const shapedAge = clamp01(linearAge * contrast + bias);
  const blurAmount = clamp01(blur);
  return shapedAge + (linearAge - shapedAge) * blurAmount;
}

/** Creates a broadly compatible GLB with every displayed sphere baked into one mesh. */
export async function createGlbBlob(data: ExportInstanceData): Promise<Blob> {
  assertExportData(data);
  const scene = await createExpandedExportScene(data);
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
    '# 260716_DLACoral\n',
    'o DLACoral\n',
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

async function createExpandedExportScene(data: ExportInstanceData): Promise<Scene> {
  const scene = new Scene();
  const geometry = await createExpandedGeometry(data);
  const material = new MeshStandardMaterial({
    color: 0xffffff,
    roughness: clamp01(data.materialRoughness),
    metalness: 0,
    flatShading: false,
    vertexColors: true,
  });
  material.name = 'DLA Age Gradient';

  const mesh = new Mesh(geometry, material);
  mesh.name = '260716_DLACoral';
  mesh.updateMatrixWorld(true);
  scene.add(mesh);
  return scene;
}

async function createExpandedGeometry(data: ExportInstanceData): Promise<BufferGeometry> {
  const vertexCount = data.spherePositions.length / 3;
  const positions = new Float32Array(data.count * data.spherePositions.length);
  const normals = new Float32Array(data.count * data.sphereNormals.length);
  const colors = new Float32Array(data.count * data.spherePositions.length);
  const particleColors = createAgeGradientColors(data);
  const instancesPerBatch = Math.max(1, Math.floor(OBJ_ASYNC_VERTEX_BUDGET / vertexCount));
  const objectMatrix = createObjectMatrix(data);
  const instanceMatrix = new Matrix4();
  const worldMatrix = new Matrix4();
  const normalMatrix = new Matrix3();
  const vertex = new Vector3();
  const normal = new Vector3();

  for (let instanceIndex = 0; instanceIndex < data.count; instanceIndex++) {
    instanceMatrix.fromArray(data.matrices, instanceIndex * 16);
    worldMatrix.multiplyMatrices(objectMatrix, instanceMatrix);
    normalMatrix.getNormalMatrix(worldMatrix);
    const colorOffset = instanceIndex * 3;
    const targetBase = instanceIndex * data.spherePositions.length;

    for (let vertexIndex = 0; vertexIndex < vertexCount; vertexIndex++) {
      const sourceOffset = vertexIndex * 3;
      const targetOffset = targetBase + sourceOffset;
      vertex
        .fromArray(data.spherePositions, sourceOffset)
        .applyMatrix4(worldMatrix);
      normal
        .fromArray(data.sphereNormals, sourceOffset)
        .applyMatrix3(normalMatrix)
        .normalize();
      positions[targetOffset] = vertex.x;
      positions[targetOffset + 1] = vertex.y;
      positions[targetOffset + 2] = vertex.z;
      normals[targetOffset] = normal.x;
      normals[targetOffset + 1] = normal.y;
      normals[targetOffset + 2] = normal.z;
      colors[targetOffset] = particleColors[colorOffset];
      colors[targetOffset + 1] = particleColors[colorOffset + 1];
      colors[targetOffset + 2] = particleColors[colorOffset + 2];
    }

    if ((instanceIndex + 1) % instancesPerBatch === 0) {
      await yieldToBrowser();
    }
  }

  const geometry = new BufferGeometry();
  geometry.setAttribute('position', new BufferAttribute(positions, 3));
  geometry.setAttribute('normal', new BufferAttribute(normals, 3));
  geometry.setAttribute('color', new BufferAttribute(colors, 3));
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  return geometry;
}

function createObjectMatrix(data: ExportInstanceData): Matrix4 {
  return new Matrix4().compose(
    new Vector3(),
    new Quaternion().setFromAxisAngle(new Vector3(0, 1, 0), data.seedRotationDegrees * DEG_TO_RAD),
    new Vector3(1, 1, 1),
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
  });
}
