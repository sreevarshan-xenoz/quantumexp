import React, { useRef, useEffect, useState } from 'react';
import * as THREE from 'three';

const ThreeJSVisualizer = ({ data, predictions, title = "3D Decision Boundary" }) => {
  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  const rendererRef = useRef(null);
  const cameraRef = useRef(null);
  const frameRef = useRef(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (!data || !mountRef.current) return;

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf8fafc);
    sceneRef.current = scene;

    // Camera setup
    const camera = new THREE.PerspectiveCamera(
      75,
      mountRef.current.clientWidth / mountRef.current.clientHeight,
      0.1,
      1000
    );
    camera.position.set(5, 5, 5);
    camera.lookAt(0, 0, 0);
    cameraRef.current = camera;

    // Renderer setup
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    mountRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(10, 10, 5);
    directionalLight.castShadow = true;
    scene.add(directionalLight);

    // Create data points
    const geometry = new THREE.SphereGeometry(0.05, 16, 16);
    const material0 = new THREE.MeshLambertMaterial({ color: 0x3b82f6 });
    const material1 = new THREE.MeshLambertMaterial({ color: 0xef4444 });

    data.forEach((point, index) => {
      const sphere = new THREE.Mesh(
        geometry,
        point[2] === 0 ? material0 : material1
      );
      sphere.position.set(point[0] * 2, point[1] * 2, 0);
      sphere.castShadow = true;
      scene.add(sphere);
    });

    // Create decision boundary surface if predictions available
    if (predictions) {
      createDecisionSurface(scene, predictions);
    }

    // Controls (basic rotation)
    let mouseX = 0;
    let mouseY = 0;
    let targetRotationX = 0;
    let targetRotationY = 0;

    const onMouseMove = (event) => {
      mouseX = (event.clientX - window.innerWidth / 2) * 0.001;
      mouseY = (event.clientY - window.innerHeight / 2) * 0.001;
      targetRotationX = mouseY;
      targetRotationY = mouseX;
    };

    mountRef.current.addEventListener('mousemove', onMouseMove);

    // Animation loop
    const animate = () => {
      frameRef.current = requestAnimationFrame(animate);

      // Smooth rotation
      scene.rotation.x += (targetRotationX - scene.rotation.x) * 0.05;
      scene.rotation.y += (targetRotationY - scene.rotation.y) * 0.05;

      renderer.render(scene, camera);
    };

    animate();
    setIsLoading(false);

    // Cleanup
    return () => {
      if (frameRef.current) {
        cancelAnimationFrame(frameRef.current);
      }
      if (mountRef.current && renderer.domElement) {
        mountRef.current.removeChild(renderer.domElement);
      }
      if (mountRef.current) {
        mountRef.current.removeEventListener('mousemove', onMouseMove);
      }
      renderer.dispose();
    };
  }, [data, predictions]);

  const createDecisionSurface = (scene, predictions) => {
    const resolution = 50;
    const geometry = new THREE.PlaneGeometry(4, 4, resolution - 1, resolution - 1);
    
    // Create height map from predictions
    const vertices = geometry.attributes.position.array;
    for (let i = 0; i < vertices.length; i += 3) {
      const x = vertices[i];
      const y = vertices[i + 1];
      
      // Map to prediction grid
      const gridX = Math.floor(((x + 2) / 4) * resolution);
      const gridY = Math.floor(((y + 2) / 4) * resolution);
      const index = Math.min(gridY * resolution + gridX, predictions.length - 1);
      
      vertices[i + 2] = predictions[index] * 0.5; // Height based on prediction
    }

    geometry.attributes.position.needsUpdate = true;
    geometry.computeVertexNormals();

    const material = new THREE.MeshLambertMaterial({
      color: 0x8b5cf6,
      transparent: true,
      opacity: 0.6,
      side: THREE.DoubleSide
    });

    const surface = new THREE.Mesh(geometry, material);
    surface.receiveShadow = true;
    scene.add(surface);
  };

  const handleResize = () => {
    if (!mountRef.current || !cameraRef.current || !rendererRef.current) return;

    const width = mountRef.current.clientWidth;
    const height = mountRef.current.clientHeight;

    cameraRef.current.aspect = width / height;
    cameraRef.current.updateProjectionMatrix();
    rendererRef.current.setSize(width, height);
  };

  useEffect(() => {
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return (
    <div className="relative w-full h-96 bg-gray-50 dark:bg-gray-800 rounded-lg overflow-hidden">
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-100 dark:bg-gray-700">
          <div className="text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-2"></div>
            <p className="text-sm text-gray-600 dark:text-gray-400">Loading 3D visualization...</p>
          </div>
        </div>
      )}
      <div ref={mountRef} className="w-full h-full" />
      <div className="absolute top-2 left-2 bg-black bg-opacity-50 text-white px-2 py-1 rounded text-sm">
        {title}
      </div>
      <div className="absolute bottom-2 right-2 text-xs text-gray-500 dark:text-gray-400">
        Mouse to rotate
      </div>
    </div>
  );
};

export default ThreeJSVisualizer;