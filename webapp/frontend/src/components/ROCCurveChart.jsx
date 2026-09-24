import { useEffect, useRef } from 'react';

const ROCCurveChart = ({ toxicity, rocData }) => {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!rocData || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // Set canvas size
    const width = canvas.width;
    const height = canvas.height;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Colors
    const colors = {
      'RF': '#10b981',      // green
      'XGBoost': '#3b82f6',  // blue
      'SVM': '#f59e0b',      // orange
      'NN': '#8b5cf6',       // purple
      'KNN': '#ec4899'       // pink
    };
    
    // Margins
    const margin = { top: 40, right: 150, bottom: 60, left: 60 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    
    // Calculate ROC curves for each model
    const rocCurves = {};
    const aucScores = {};
    
    Object.keys(rocData).forEach(model => {
      const { probabilities, labels } = rocData[model];
      
      // Sort by probability
      const sortedData = probabilities.map((prob, i) => ({
        prob,
        label: labels[i]
      })).sort((a, b) => b.prob - a.prob);
      
      // Calculate ROC points
      const points = [{ fpr: 0, tpr: 0 }];
      let tp = 0, fp = 0;
      const totalPositive = labels.filter(l => l === 1).length;
      const totalNegative = labels.length - totalPositive;
      
      sortedData.forEach(({ label }) => {
        if (label === 1) tp++;
        else fp++;
        
        const tpr = tp / totalPositive;
        const fpr = fp / totalNegative;
        points.push({ fpr, tpr });
      });
      
      points.push({ fpr: 1, tpr: 1 });
      rocCurves[model] = points;
      
      // Calculate AUC using trapezoidal rule
      let auc = 0;
      for (let i = 1; i < points.length; i++) {
        const width = points[i].fpr - points[i-1].fpr;
        const height = (points[i].tpr + points[i-1].tpr) / 2;
        auc += width * height;
      }
      aucScores[model] = auc;
    });
    
    // Draw background
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(margin.left, margin.top, plotWidth, plotHeight);
    
    // Draw grid
    ctx.strokeStyle = '#2d2d44';
    ctx.lineWidth = 1;
    
    // Vertical grid lines
    for (let i = 0; i <= 10; i++) {
      const x = margin.left + (i / 10) * plotWidth;
      ctx.beginPath();
      ctx.moveTo(x, margin.top);
      ctx.lineTo(x, margin.top + plotHeight);
      ctx.stroke();
    }
    
    // Horizontal grid lines
    for (let i = 0; i <= 10; i++) {
      const y = margin.top + plotHeight - (i / 10) * plotHeight;
      ctx.beginPath();
      ctx.moveTo(margin.left, y);
      ctx.lineTo(margin.left + plotWidth, y);
      ctx.stroke();
    }
    
    // Draw diagonal reference line (random classifier)
    ctx.strokeStyle = '#6b7280';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(margin.left, margin.top + plotHeight);
    ctx.lineTo(margin.left + plotWidth, margin.top);
    ctx.stroke();
    ctx.setLineDash([]);
    
    // Draw ROC curves
    Object.keys(rocCurves).forEach(model => {
      const points = rocCurves[model];
      const color = colors[model] || '#ffffff';
      
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.beginPath();
      
      points.forEach((point, i) => {
        const x = margin.left + point.fpr * plotWidth;
        const y = margin.top + plotHeight - point.tpr * plotHeight;
        
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      
      ctx.stroke();
    });
    
    // Draw axes
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    
    // X-axis
    ctx.beginPath();
    ctx.moveTo(margin.left, margin.top + plotHeight);
    ctx.lineTo(margin.left + plotWidth, margin.top + plotHeight);
    ctx.stroke();
    
    // Y-axis
    ctx.beginPath();
    ctx.moveTo(margin.left, margin.top);
    ctx.lineTo(margin.left, margin.top + plotHeight);
    ctx.stroke();
    
    // Draw labels
    ctx.fillStyle = '#ffffff';
    ctx.font = '14px sans-serif';
    ctx.textAlign = 'center';
    
    // X-axis labels
    for (let i = 0; i <= 10; i++) {
      const x = margin.left + (i / 10) * plotWidth;
      const y = margin.top + plotHeight + 30;
      ctx.fillText((i / 10).toFixed(1), x, y);
    }
    
    // Y-axis labels
    ctx.textAlign = 'right';
    for (let i = 0; i <= 10; i++) {
      const x = margin.left - 10;
      const y = margin.top + plotHeight - (i / 10) * plotHeight + 5;
      ctx.fillText((i / 10).toFixed(1), x, y);
    }
    
    // Axis titles
    ctx.font = 'bold 16px sans-serif';
    ctx.textAlign = 'center';
    
    // X-axis title
    ctx.fillText('False Positive Rate', margin.left + plotWidth / 2, height - 10);
    
    // Y-axis title
    ctx.save();
    ctx.translate(15, margin.top + plotHeight / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('True Positive Rate', 0, 0);
    ctx.restore();
    
    // Title
    ctx.font = 'bold 18px sans-serif';
    ctx.fillText(`ROC Curves - ${toxicity}`, margin.left + plotWidth / 2, 25);
    
    // Legend
    const legendX = width - margin.right + 10;
    let legendY = margin.top;
    
    ctx.font = 'bold 14px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('Models:', legendX, legendY);
    legendY += 25;
    
    ctx.font = '12px sans-serif';
    Object.keys(rocCurves).forEach(model => {
      const color = colors[model] || '#ffffff';
      const auc = aucScores[model];
      
      // Draw colored line
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(legendX, legendY - 5);
      ctx.lineTo(legendX + 25, legendY - 5);
      ctx.stroke();
      
      // Draw text
      ctx.fillStyle = '#ffffff';
      ctx.fillText(`${model} (AUC: ${auc.toFixed(3)})`, legendX + 35, legendY);
      legendY += 25;
    });
    
    // Random classifier legend
    legendY += 10;
    ctx.strokeStyle = '#6b7280';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(legendX, legendY - 5);
    ctx.lineTo(legendX + 25, legendY - 5);
    ctx.stroke();
    ctx.setLineDash([]);
    
    ctx.fillStyle = '#9ca3af';
    ctx.fillText('Random (AUC: 0.500)', legendX + 35, legendY);
    
  }, [toxicity, rocData]);

  return (
    <div className="glass-panel p-6">
      <div className="overflow-x-auto">
        <canvas
          ref={canvasRef}
          width={900}
          height={600}
          className="mx-auto"
          style={{ maxWidth: '100%', height: 'auto' }}
        />
      </div>
    </div>
  );
};

export default ROCCurveChart;
