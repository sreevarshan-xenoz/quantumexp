import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

const DatasetVisualizer = ({ data, datasetType, isLoading = false }) => {
  const svgRef = useRef(null);

  useEffect(() => {
    if (!data || isLoading) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const width = 500;
    const height = 350;
    const margin = { top: 20, right: 20, bottom: 40, left: 40 };

    // Set up scales
    const xExtent = d3.extent(data, d => d[0]);
    const yExtent = d3.extent(data, d => d[1]);
    
    const x = d3.scaleLinear()
      .domain([xExtent[0] - 0.5, xExtent[1] + 0.5])
      .range([margin.left, width - margin.right]);

    const y = d3.scaleLinear()
      .domain([yExtent[0] - 0.5, yExtent[1] + 0.5])
      .range([height - margin.bottom, margin.top]);

    const color = d3.scaleOrdinal()
      .domain([0, 1])
      .range(["#3b82f6", "#ef4444"]);

    // Add background grid
    const xTicks = x.ticks(8);
    const yTicks = y.ticks(6);

    svg.append("g")
      .selectAll("line")
      .data(xTicks)
      .join("line")
      .attr("x1", d => x(d))
      .attr("x2", d => x(d))
      .attr("y1", margin.top)
      .attr("y2", height - margin.bottom)
      .attr("stroke", "#e5e7eb")
      .attr("stroke-width", 0.5);

    svg.append("g")
      .selectAll("line")
      .data(yTicks)
      .join("line")
      .attr("x1", margin.left)
      .attr("x2", width - margin.right)
      .attr("y1", d => y(d))
      .attr("y2", d => y(d))
      .attr("stroke", "#e5e7eb")
      .attr("stroke-width", 0.5);

    // Add data points with animation
    svg.append("g")
      .selectAll("circle")
      .data(data)
      .join("circle")
      .attr("cx", d => x(d[0]))
      .attr("cy", d => y(d[1]))
      .attr("r", 0)
      .attr("fill", d => color(d[2]))
      .attr("opacity", 0.8)
      .attr("stroke", "white")
      .attr("stroke-width", 1)
      .transition()
      .duration(800)
      .delay((d, i) => i * 2)
      .attr("r", 4);

    // Add axes
    svg.append("g")
      .attr("transform", `translate(0,${height - margin.bottom})`)
      .call(d3.axisBottom(x).ticks(8))
      .selectAll("text")
      .style("font-size", "12px")
      .style("fill", "#6b7280");

    svg.append("g")
      .attr("transform", `translate(${margin.left},0)`)
      .call(d3.axisLeft(y).ticks(6))
      .selectAll("text")
      .style("font-size", "12px")
      .style("fill", "#6b7280");

    // Add axis labels
    svg.append("text")
      .attr("x", width / 2)
      .attr("y", height - 5)
      .attr("text-anchor", "middle")
      .style("font-size", "14px")
      .style("fill", "#374151")
      .text("Feature 1");

    svg.append("text")
      .attr("transform", "rotate(-90)")
      .attr("x", -height / 2)
      .attr("y", 15)
      .attr("text-anchor", "middle")
      .style("font-size", "14px")
      .style("fill", "#374151")
      .text("Feature 2");

    // Add title
    svg.append("text")
      .attr("x", width / 2)
      .attr("y", 15)
      .attr("text-anchor", "middle")
      .style("font-size", "16px")
      .style("font-weight", "600")
      .style("fill", "#1f2937")
      .text(`${datasetType.charAt(0).toUpperCase() + datasetType.slice(1)} Dataset`);

    // Add legend
    const legend = svg.append("g")
      .attr("transform", `translate(${width - 100}, 30)`);

    legend.append("circle")
      .attr("cx", 0)
      .attr("cy", 0)
      .attr("r", 6)
      .attr("fill", "#3b82f6");

    legend.append("text")
      .attr("x", 12)
      .attr("y", 5)
      .style("font-size", "12px")
      .style("fill", "#374151")
      .text("Class 0");

    legend.append("circle")
      .attr("cx", 0)
      .attr("cy", 20)
      .attr("r", 6)
      .attr("fill", "#ef4444");

    legend.append("text")
      .attr("x", 12)
      .attr("y", 25)
      .style("font-size", "12px")
      .style("fill", "#374151")
      .text("Class 1");

  }, [data, datasetType, isLoading]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-80 bg-gray-50 dark:bg-gray-800 rounded-lg">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600 dark:text-gray-400">Generating dataset...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="dataset-visualizer bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm">
      <svg 
        ref={svgRef} 
        width="100%" 
        height="350"
        viewBox="0 0 500 350"
        className="max-w-full h-auto"
      />
    </div>
  );
};

export default DatasetVisualizer;