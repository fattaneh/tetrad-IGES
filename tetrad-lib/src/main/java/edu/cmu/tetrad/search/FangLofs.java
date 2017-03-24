///////////////////////////////////////////////////////////////////////////////
// For information as to what this class does, see the Javadoc, below.       //
// Copyright (C) 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006,       //
// 2007, 2008, 2009, 2010, 2014, 2015 by Peter Spirtes, Richard Scheines, Joseph   //
// Ramsey, and Clark Glymour.                                                //
//                                                                           //
// This program is free software; you can redistribute it and/or modify      //
// it under the terms of the GNU General Public License as published by      //
// the Free Software Foundation; either version 2 of the License, or         //
// (at your option) any later version.                                       //
//                                                                           //
// This program is distributed in the hope that it will be useful,           //
// but WITHOUT ANY WARRANTY; without even the implied warranty of            //
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             //
// GNU General Public License for more details.                              //
//                                                                           //
// You should have received a copy of the GNU General Public License         //
// along with this program; if not, write to the Free Software               //
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA //
///////////////////////////////////////////////////////////////////////////////

package edu.cmu.tetrad.search;

import edu.cmu.tetrad.data.*;
import edu.cmu.tetrad.graph.*;
import edu.cmu.tetrad.regression.RegressionDataset;

import java.awt.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static edu.cmu.tetrad.util.StatUtils.correlation;
import static java.lang.Math.abs;
import static java.lang.Math.signum;

/**
 * Fast adjacency search followed by robust skew orientation. Checks are done for adding
 * two-cycles. The two-cycle checks do not require non-Gaussianity. The robust skew
 * orientation of edges left or right does.
 *
 * @author Joseph Ramsey
 */
public final class FangLofs implements GraphSearch {

    private final Lofs2.Rule rule;
    // Elapsed time of the search, in milliseconds.
    private long elapsed = 0;

    // The data sets being analyzed. They must all have the same variables and the same
    // number of records.
    private List<DataSet> dataSets = null;

    // For the Fast Adjacency Search.
    private int depth = -1;

    // For the SEM BIC score, for the Fast Adjacency Search.
    private double penaltyDiscount = 1;

    // Knowledge the the search will obey, of forbidden and required edges.
    private IKnowledge knowledge = new Knowledge2();

    // The maximum coefficient in absolute value (used for orienting 2-cycles.
    private double maxCoef = 0.6;

    // Alpha level for detecting dependent errors.
    private double correlatedErrorsAlpha;

    // True if dependent residuals should be marked with o-o edges.
    private boolean markDependentResidualsInGraph = false;

    /**
     * @param dataSets These datasets must all have the same variables, in the same order.
     */
    public FangLofs(List<DataSet> dataSets, Lofs2.Rule rule) {
        this.dataSets = dataSets;
        this.rule = rule;
    }

    //======================================== PUBLIC METHODS ====================================//

    /**
     * Runs the search on the concatenated data, returning a graph, possibly cyclic, possibly with
     * two-cycles. Runs the fast adjacency search (FAS, Spirtes et al., 2000) follows by a modification
     * of the robust skew rule (Pairwise Likelihood Ratios for Estimation of Non-Gaussian Structural
     * Equation Models, Smith and Hyvarinen), together with some heuristics for orienting two-cycles.
     *
     * @return the graph. Some of the edges may be undirected (though it shouldn't be many in most cases)
     * and some of the adjacencies may be two-cycles.
     */
    public Graph search() {
        long start = System.currentTimeMillis();

        List<DataSet> _dataSets = new ArrayList<>();
        for (DataSet dataSet : dataSets) _dataSets.add(DataUtils.standardizeData(dataSet));

        DataSet dataSet = DataUtils.concatenate(_dataSets);

        SemBicScore score = new SemBicScore(new CovarianceMatrix(dataSet));
        score.setPenaltyDiscount(penaltyDiscount);
        IndependenceTest test = new IndTestScore(score, dataSet);
        List<Node> variables = dataSet.getVariables();

        double[][] colData = dataSet.getDoubleData().transpose().toArray();

        System.out.println("FAS");

        Fas fas = new Fas(test);
        fas.setDepth(getDepth());
        fas.setVerbose(false);
        fas.setKnowledge(knowledge);
        Graph G0 = fas.search();

        Lofs2 lofs2 = new Lofs2(G0, Collections.singletonList(dataSet));
        lofs2.setRule(rule);
        lofs2.setKnowledge(knowledge);
        Graph graph = lofs2.orient();

        System.out.println("Done");

        long stop = System.currentTimeMillis();
        this.elapsed = stop - start;

        return graph;
    }

    /**
     * @return The depth of search for the Fast Adjacency Search (FAS).
     */
    public int getDepth() {
        return depth;
    }

    /**
     * @param depth The depth of search for the Fast Adjacency Search (S). The default is -1.
     *              unlimited. Making this too high may results in statistical errors.
     */
    public void setDepth(int depth) {
        this.depth = depth;
    }

    /**
     * @return The elapsed time in milliseconds.
     */
    public long getElapsedTime() {
        return elapsed;
    }

    /**
     * @return Returns the penalty discount used for the adjacency search. The default is 1,
     * though a higher value is recommended, say, 2, 3, or 4.
     */
    public double getPenaltyDiscount() {
        return penaltyDiscount;
    }

    /**
     * @param penaltyDiscount Sets the penalty discount used for the adjacency search.
     *                        The default is 1, though a higher value is recommended, say,
     *                        2, 3, or 4.
     */
    public void setPenaltyDiscount(double penaltyDiscount) {
        this.penaltyDiscount = penaltyDiscount;
    }

    /**
     * @return The maximum coefficient in absoluate value (used for orienting 2-cycles).
     */
    public double getMaxCoef() {
        return maxCoef;
    }

    /**
     * @param maxCoef The maximum coefficient in absoluate value (used for orienting 2-cycles).f
     */
    public void setMaxCoef(double maxCoef) {
        this.maxCoef = maxCoef;
    }

    /**
     * @return Alpha level for detecting dependent errors. The lower this is set, the fewer dependent
     * errors will be found.
     */
    public double getCorrelatedErrorsAlpha() {
        return correlatedErrorsAlpha;
    }

    /**
     * @param correlatedErrorsAlpha Alpha level for detecting dependent errors. The lower this is set, the fewer
     *                       dependent errors will be found.
     */
    public void setCorrelatedErrorsAlpha(double correlatedErrorsAlpha) {
        this.correlatedErrorsAlpha = correlatedErrorsAlpha;
    }

    /**
     * @return True if dependent residuals should be marked in the graph with o-o dark green edges.
     */
    public boolean isMarkDependentResidualsInGraph() {
        return markDependentResidualsInGraph;
    }

    /**
     * @param markDependentResidualsInGraph True if dependent residuals should be marked in the
     *                                      graph with o-o dark green edges.
     */
    public void setMarkDependentResidualsInGraph(boolean markDependentResidualsInGraph) {
        this.markDependentResidualsInGraph = markDependentResidualsInGraph;
    }

    /**
     * @return the current knowledge.
     */
    public IKnowledge getKnowledge() {
        return knowledge;
    }

    /**
     * @param knowledge Knowledge of forbidden and required edges.
     */
    public void setKnowledge(IKnowledge knowledge) {
        this.knowledge = knowledge;
    }

    //======================================== PRIVATE METHODS ====================================//

    private boolean knowledgeOrients(Node left, Node right) {
        return knowledge.isForbidden(right.getName(), left.getName()) || knowledge.isRequired(left.getName(), right.getName());
    }

}






