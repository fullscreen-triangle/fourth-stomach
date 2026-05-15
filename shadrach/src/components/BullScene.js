import { Canvas, useFrame } from '@react-three/fiber';
import { useGLTF, useAnimations, OrbitControls } from '@react-three/drei';
import { useEffect, useRef, useState, useCallback } from 'react';

function BullModel({ onInfo }) {
  const group = useRef();
  const { scene, animations } = useGLTF('/models/maquina_bull.glb');
  const { actions, names } = useAnimations(animations, group);
  const idxRef     = useRef(0);
  const currentRef = useRef(null);

  const playAnim = useCallback((idx) => {
    if (!names.length) return;
    const name = names[idx];
    if (!name || !actions[name]) return;
    if (currentRef.current) currentRef.current.fadeOut(0.4);
    const action = actions[name];
    action.reset().fadeIn(0.4).play();
    currentRef.current = action;
    idxRef.current = idx;
    onInfo({ name, idx, total: names.length });
  }, [actions, names, onInfo]);

  // Mount: play first animation, then auto-cycle
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

  // Slow idle sway when no animation is active
  useFrame(({ clock }) => {
    if (!group.current) return;
    if (!names.length) {
      group.current.rotation.y = Math.sin(clock.elapsedTime * 0.18) * 0.35;
    }
  });

  return <primitive ref={group} object={scene} />;
}

export default function BullScene() {
  const [info, setInfo] = useState({ name: '', idx: 0, total: 0 });

  return (
    <div className="w-full h-full relative" style={{ background: '#0a0e17' }}>
      <Canvas
        camera={{ position: [0, 1.2, 6], fov: 42 }}
        gl={{ antialias: true }}
        shadows
      >
        {/* Dark atmospheric lighting — teal key, gold rim, soft fill */}
        <ambientLight intensity={0.15} />
        <directionalLight
          position={[4, 8, 4]}
          intensity={2.2}
          color="#2ca89a"
          castShadow
        />
        <directionalLight
          position={[-5, 1, -4]}
          intensity={1.0}
          color="#d4a843"
        />
        <pointLight position={[0, 4, 3]} intensity={0.6} color="#ffffff" />
        <hemisphereLight args={['#0a0e17', '#1a3a5c', 0.6]} />

        <BullModel onInfo={setInfo} />

        {/* Allow gentle orbit, no zoom */}
        <OrbitControls
          enableZoom={false}
          enablePan={false}
          minPolarAngle={Math.PI / 4}
          maxPolarAngle={Math.PI / 1.8}
          autoRotate={false}
          dampingFactor={0.05}
          enableDamping
        />
      </Canvas>

      {/* Animation indicator */}
      {info.total > 0 && (
        <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 pointer-events-none">
          {/* Dot pagination */}
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
          {/* Animation name */}
          <span className="text-primary/30 font-mono text-xs tracking-[0.3em] uppercase">
            {info.name}
          </span>
        </div>
      )}
    </div>
  );
}

useGLTF.preload('/models/maquina_bull.glb');
