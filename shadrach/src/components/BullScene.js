import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { useGLTF, useAnimations, OrbitControls } from '@react-three/drei';
import { useEffect, useRef, useState, useCallback } from 'react';
import * as THREE from 'three';

function BullModel({ onInfo }) {
  const group = useRef();
  const { scene, animations } = useGLTF('/models/maquina_bull.glb');
  const { actions, names }    = useAnimations(animations, group);
  const { camera }            = useThree();
  const idxRef                = useRef(0);
  const currentRef            = useRef(null);

  // Auto-fit camera to the model's bounding box after load
  useEffect(() => {
    if (!scene) return;

    const box = new THREE.Box3().setFromObject(scene);
    if (box.isEmpty()) return;

    const size   = new THREE.Vector3();
    const center = new THREE.Vector3();
    box.getSize(size);
    box.getCenter(center);

    // Translate model so its centre is at world origin
    scene.position.sub(center);

    // Place camera so the model fills ~70 % of the vertical FOV
    const maxDim = Math.max(size.x, size.y, size.z);
    const fovRad = (camera.fov * Math.PI) / 180;
    const dist   = (maxDim / 2 / Math.tan(fovRad / 2)) * 1.6;

    camera.position.set(0, size.y * 0.05, dist);
    camera.near = dist * 0.01;
    camera.far  = dist * 200;
    camera.lookAt(0, 0, 0);
    camera.updateProjectionMatrix();
  }, [scene, camera]);

  const playAnim = useCallback((idx) => {
    if (!names.length) return;
    const name = names[idx];
    if (!name || !actions[name]) return;
    if (currentRef.current) currentRef.current.fadeOut(0.5);
    const action = actions[name];
    action.reset().fadeIn(0.5).play();
    currentRef.current = action;
    idxRef.current     = idx;
    onInfo({ name, idx, total: names.length });
  }, [actions, names, onInfo]);

  useEffect(() => {
    if (!names.length) {
      onInfo({ name: '', idx: 0, total: 0 });
      return;
    }
    playAnim(0);
    if (names.length <= 1) return;
    const id = setInterval(() => {
      playAnim((idxRef.current + 1) % names.length);
    }, 7000);
    return () => clearInterval(id);
  }, [names.length, playAnim]); // eslint-disable-line react-hooks/exhaustive-deps

  // Gentle idle sway when no GLB animations exist
  useFrame(({ clock }) => {
    if (!group.current || names.length) return;
    group.current.rotation.y = Math.sin(clock.elapsedTime * 0.18) * 0.4;
  });

  return <primitive ref={group} object={scene} />;
}

export default function BullScene() {
  const [info, setInfo] = useState({ name: '', idx: 0, total: 0 });

  return (
    <div className="w-full h-full relative" style={{ background: '#0a0e17' }}>
      <Canvas
        frameloop="always"
        camera={{ position: [0, 0, 10], fov: 42 }}
        gl={{ antialias: true }}
      >
        {/* Lighting — teal key from above-front, gold rim from behind, soft fill */}
        <ambientLight intensity={0.5} />
        <directionalLight position={[3, 6, 4]}  intensity={3.0} color="#2ca89a" />
        <directionalLight position={[-4, 2, -3]} intensity={1.5} color="#d4a843" />
        <pointLight position={[0, 3, 2]} intensity={2.0} color="#ffffff" />
        <hemisphereLight args={['#1a3a5c', '#0a0e17', 1.0]} />

        <BullModel onInfo={setInfo} />

        <OrbitControls
          enableZoom={false}
          enablePan={false}
          minPolarAngle={Math.PI / 4}
          maxPolarAngle={Math.PI / 1.7}
          enableDamping
          dampingFactor={0.06}
        />
      </Canvas>

      {/* Animation pagination indicator */}
      {info.total > 0 && (
        <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 pointer-events-none">
          <div className="flex gap-2">
            {Array.from({ length: info.total }, (_, i) => (
              <span
                key={i}
                className={`block rounded-full transition-all duration-500 ${
                  i === info.idx
                    ? 'w-4 h-1.5 bg-primary'
                    : 'w-1.5 h-1.5 bg-primary/20'
                }`}
              />
            ))}
          </div>
          <span className="text-primary/30 font-mono text-xs tracking-[0.3em] uppercase">
            {info.name}
          </span>
        </div>
      )}
    </div>
  );
}

useGLTF.preload('/models/maquina_bull.glb');
