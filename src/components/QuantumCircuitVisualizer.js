import React, { useEffect, useRef } from 'react';

const QuantumCircuitVisualizer = ({ model, featureMap = 'zz' }) => {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    
    // Set canvas size
    canvas.width = 600 * dpr;
    canvas.height = 200 * dpr;
    canvas.style.width = '600px';
    canvas.style.height = '200px';
    ctx.scale(dpr, dpr);

    // Clear canvas
    ctx.clearRect(0, 0, 600, 200);

    // Draw circuit based on feature map type
    if (featureMap === 'zz') {
      drawZZCircuit(ctx);
    } else if (featureMap === 'z') {
      drawZCircuit(ctx);
    } else if (featureMap === 'pauli') {
      drawPauliCircuit(ctx);
    } else {
      drawZZCircuit(ctx);
    }
  }, [model, featureMap]);

  const drawZZCircuit = (ctx) => {
    const width = 600;
    const height = 200;
    const qubitY = [60, 140];
    
    // Draw title
    ctx.fillStyle = '#1f2937';
    ctx.font = 'bold 16px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('ZZ Feature Map Circuit', width / 2, 25);

    // Draw wires
    ctx.strokeStyle = '#374151';
    ctx.lineWidth = 2;
    qubitY.forEach(y => {
      ctx.beginPath();
      ctx.moveTo(50, y);
      ctx.lineTo(width - 50, y);
      ctx.stroke();
    });

    // Draw qubit labels
    ctx.fillStyle = '#3b82f6';
    ctx.font = '14px Arial';
    ctx.textAlign = 'center';
    qubitY.forEach((y, i) => {
      ctx.beginPath();
      ctx.arc(30, y, 12, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = 'white';
      ctx.fillText(`q${i}`, 30, y + 4);
      ctx.fillStyle = '#3b82f6';
    });

    // Draw gates for 3 repetitions
    for (let rep = 0; rep < 3; rep++) {
      const x = 120 + rep * 150;
      
      // H gates
      qubitY.forEach(y => {
        ctx.fillStyle = '#10b981';
        ctx.fillRect(x - 15, y - 15, 30, 30);
        ctx.strokeStyle = '#065f46';
        ctx.lineWidth = 1;
        ctx.strokeRect(x - 15, y - 15, 30, 30);
        ctx.fillStyle = 'white';
        ctx.font = 'bold 14px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('H', x, y + 4);
      });

      // ZZ entangling gate
      const zzX = x + 60;
      ctx.strokeStyle = '#8b5cf6';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(zzX, qubitY[0]);
      ctx.lineTo(zzX, qubitY[1]);
      ctx.stroke();

      // Control points
      qubitY.forEach(y => {
        ctx.fillStyle = '#8b5cf6';
        ctx.beginPath();
        ctx.arc(zzX, y, 8, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = 'white';
        ctx.font = 'bold 10px Arial';
        ctx.fillText('Z', zzX, y + 3);
      });

      // Parameter labels
      ctx.fillStyle = '#6b7280';
      ctx.font = '12px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(`φ${rep + 1}`, x, qubitY[0] - 25);
      ctx.fillText(`θ${rep + 1}`, zzX, qubitY[0] - 25);
    }

    // Add measurement
    const measX = width - 80;
    qubitY.forEach((y, i) => {
      ctx.fillStyle = '#f59e0b';
      ctx.fillRect(measX - 20, y - 12, 40, 24);
      ctx.strokeStyle = '#d97706';
      ctx.lineWidth = 1;
      ctx.strokeRect(measX - 20, y - 12, 40, 24);
      ctx.fillStyle = 'white';
      ctx.font = 'bold 12px Arial';
      ctx.textAlign = 'center';
      ctx.fillText('M', measX, y + 4);
    });
  };

  const drawZCircuit = (ctx) => {
    const width = 600;
    const height = 200;
    const qubitY = [60, 140];
    
    // Draw title
    ctx.fillStyle = '#1f2937';
    ctx.font = 'bold 16px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Z Feature Map Circuit', width / 2, 25);

    // Draw wires
    ctx.strokeStyle = '#374151';
    ctx.lineWidth = 2;
    qubitY.forEach(y => {
      ctx.beginPath();
      ctx.moveTo(50, y);
      ctx.lineTo(width - 50, y);
      ctx.stroke();
    });

    // Draw qubit labels
    ctx.fillStyle = '#3b82f6';
    ctx.font = '14px Arial';
    ctx.textAlign = 'center';
    qubitY.forEach((y, i) => {
      ctx.beginPath();
      ctx.arc(30, y, 12, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = 'white';
      ctx.fillText(`q${i}`, 30, y + 4);
      ctx.fillStyle = '#3b82f6';
    });

    // Draw Z rotation gates
    for (let rep = 0; rep < 3; rep++) {
      const x = 120 + rep * 120;
      
      qubitY.forEach((y, i) => {
        ctx.fillStyle = '#8b5cf6';
        ctx.fillRect(x - 20, y - 15, 40, 30);
        ctx.strokeStyle = '#6d28d9';
        ctx.lineWidth = 1;
        ctx.strokeRect(x - 20, y - 15, 40, 30);
        ctx.fillStyle = 'white';
        ctx.font = 'bold 12px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(`RZ(φ${i})`, x, y + 4);
      });
    }
  };

  const drawPauliCircuit = (ctx) => {
    const width = 600;
    const height = 200;
    const qubitY = [60, 140];
    
    // Draw title
    ctx.fillStyle = '#1f2937';
    ctx.font = 'bold 16px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Pauli Feature Map Circuit', width / 2, 25);

    // Draw wires
    ctx.strokeStyle = '#374151';
    ctx.lineWidth = 2;
    qubitY.forEach(y => {
      ctx.beginPath();
      ctx.moveTo(50, y);
      ctx.lineTo(width - 50, y);
      ctx.stroke();
    });

    // Draw qubit labels
    ctx.fillStyle = '#3b82f6';
    ctx.font = '14px Arial';
    ctx.textAlign = 'center';
    qubitY.forEach((y, i) => {
      ctx.beginPath();
      ctx.arc(30, y, 12, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = 'white';
      ctx.fillText(`q${i}`, 30, y + 4);
      ctx.fillStyle = '#3b82f6';
    });

    // Draw Pauli gates
    const pauliGates = ['X', 'Y', 'Z'];
    for (let rep = 0; rep < 2; rep++) {
      const x = 120 + rep * 180;
      
      qubitY.forEach((y, i) => {
        // Single qubit Pauli gates
        ctx.fillStyle = '#ef4444';
        ctx.fillRect(x - 15, y - 15, 30, 30);
        ctx.strokeStyle = '#dc2626';
        ctx.lineWidth = 1;
        ctx.strokeRect(x - 15, y - 15, 30, 30);
        ctx.fillStyle = 'white';
        ctx.font = 'bold 14px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(pauliGates[i], x, y + 4);
      });

      // Two-qubit Pauli gate (XX)
      const xxX = x + 80;
      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(xxX, qubitY[0]);
      ctx.lineTo(xxX, qubitY[1]);
      ctx.stroke();

      qubitY.forEach(y => {
        ctx.fillStyle = '#f59e0b';
        ctx.fillRect(xxX - 15, y - 15, 30, 30);
        ctx.strokeStyle = '#d97706';
        ctx.lineWidth = 1;
        ctx.strokeRect(xxX - 15, y - 15, 30, 30);
        ctx.fillStyle = 'white';
        ctx.font = 'bold 14px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('X', xxX, y + 4);
      });
    }
  };

  return (
    <div className="quantum-circuit-visualizer bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm">
      <div className="flex justify-center">
        <canvas 
          ref={canvasRef} 
          className="border border-gray-200 dark:border-gray-600 rounded-lg"
        />
      </div>
      <div className="mt-4 text-center">
        <p className="text-sm text-gray-600 dark:text-gray-400">
          Interactive quantum circuit visualization for {featureMap.toUpperCase()} feature map
        </p>
      </div>
    </div>
  );
};

export default QuantumCircuitVisualizer;