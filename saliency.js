export default class saliency {
  constructor(utils, elem) {
    this.utils = utils;
    this.div = elem;
    this.node = elem.node();
    this.tooltip = d3.select("#saliency-tooltip");
    this.layers = {
      interactions: null,
      drawing: null,
    };
    this.margin = {
      t: 8,
      r: 8,
      b: 8,
      l: 72,
    };
    this.data = null;
    this.controls = {
      span: "token",
      score: "content",
      colorBy: "diff_true",
    };
  }

  init() {
    const self = this;

    const div = self.div;
    const margin = self.margin;

    // add groups in layer order (i.e., draw element groups in this order)
    const interactionsLayer = div.append("div").attr("class", "interaction-layer");
    const drawingLayer = interactionsLayer.append("div").attr("class", "word-spans");

    // save groups to access later
    self.layers.interactions = interactionsLayer;
    self.layers.drawing = drawingLayer;
  }

  clear() {
    this.layers.drawing.selectAll("*").remove();
  }

  render() {
    const self = this;

    if (!self.data) return;

    const fw = parseFloat(self.div.style("width"));
    const fh = parseFloat(self.div.style("height"));

    let data;
    switch (self.controls.score) {
      case "content":
        data = self.data.saliencyScores.content_saliency_scores;
        break;
      case "wording":
        data = self.data.saliencyScores.wording_saliency_scores;
        break;
    }

    const trueValueExtent = d3.extent(data.map((x) => x.diff_true));
    const absValueExtent = d3.extent(data.map((x) => x.diff_abs));

    let colorScale;
    if (self.controls.colorBy == "diff_true") {
      const lb = trueValueExtent[0];
      const ub = trueValueExtent[1];
      let extent;
      if (lb < 0 && ub > 0) {
        extent = [trueValueExtent[0], 0, trueValueExtent[1]];
        colorScale = d3.scaleDiverging(extent, (t) => d3.interpolateRdYlGn(t));
      } else if (lb > 0 && ub > 0) {
        extent = trueValueExtent;
        colorScale = d3.scaleSequential(extent, (t) => d3.interpolateGreens(t));
      } else if (lb < 0 && ub < 0) {
        extent = trueValueExtent;
        colorScale = d3.scaleSequential(extent, (t) => d3.interpolateReds(t));
      }
    } else if (self.controls.colorBy == "diff_sum_norm" || self.controls.colorBy == "diff_max_norm") {
      colorScale = d3.scaleSequential([0, 1], d3.interpolatePurples);
    } else {
      colorScale = d3.scaleSequential(absValueExtent, d3.interpolatePurples);
    }

    self.layers.drawing
      .selectAll("div")
      .data(data)
      .join("div")
      .attr("class", (x) => {
        const leftSpace = x[self.controls.span].startsWith(" ") ? "left-space" : "";
        const rightSpace = x[self.controls.span].endsWith(" ") ? "right-space" : "";
        return `word-span ${leftSpace} ${rightSpace}`;
      })
      .style("color", "black")
      .style("border-bottom", (x) => {
        const color = colorScale(x[self.controls.colorBy]);
        return `5px solid ${color}`;
      })
      .html((x) =>
        x[self.controls.span]
          .trim()
          .replaceAll("&", "&amp;")
          .replaceAll("<", "&lt;")
          .replaceAll(">", "&gt;")
          .replaceAll('"', "&quot;")
          .replaceAll("'", "&#39;")
      )
      .on("mouseenter", mouseenter)
      .on("mousemove", mousemove)
      .on("mouseleave", mouseleave);

    // ============================= HOVER ================================= //

    function mouseenter(event, d) {
      const trueScore = self.data.scores[self.controls.score];
      const trueLB = (Math.round(trueValueExtent[0] * 100) / 100).toFixed(2);
      const trueUB = (Math.round(trueValueExtent[1] * 100) / 100).toFixed(2);
      const absLB = (Math.round(absValueExtent[0] * 100) / 100).toFixed(2);
      const absUB = (Math.round(absValueExtent[1] * 100) / 100).toFixed(2);
      let html = `
        <div>True score:&nbsp;<b>${trueScore}</b></div>
        <div>New score:&nbsp;<b>${d.score}</b></div>
        <div>Diff (true) [${trueLB}, 0, ${trueUB}]:&nbsp;<b>${d.diff_true}</b></div>
        <div>Diff (abs) [${absLB}, ${absUB}]:&nbsp;<b>${d.diff_abs}</b></div>
        <div>Diff (sum norm) [0, 1]:&nbsp;<b>${d.diff_sum_norm}</b></div>
        <div>Diff (max norm) [0, 1]:&nbsp;<b>${d.diff_max_norm}</b></div>
      `;
      self.tooltip.html(html);
      self.tooltip.style("display", "block").style("opacity", 1);
    }
    function mousemove(event, d) {
      positionTooltip(event.x, event.y);
    }
    function mouseleave(event, d) {
      self.tooltip.html("");
      self.tooltip.style("display", "none").style("opacity", 0);
    }

    // ============================= CLICK ================================= //

    // function click(event, d) {}

    // ============================= HELPER ================================ //

    const topPad = 12; // draw tooltip px up from top edge of cursor
    const leftPad = -12; // draw tooltip px left from left edge of cursor

    function positionTooltip(eventX, eventY) {
      const vw = Math.max(document.documentElement.clientWidth || 0, window.innerWidth || 0);
      const width = self.tooltip.node().getBoundingClientRect().width + 2;
      const height = self.tooltip.node().getBoundingClientRect().height + 2;
      const left = window.scrollX + eventX + width - leftPad >= vw ? vw - width : window.scrollX + eventX - leftPad;
      const top = window.scrollY + eventY - height - topPad <= 0 ? 0 : window.scrollY + eventY - height - topPad;
      self.tooltip.style("left", `${left}px`).style("top", `${top}px`);
    }
  }
}
