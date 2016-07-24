package edu.cmu.tetrad.util;

import java.util.List;

/**
 * Tags a gadget as having parameters
 */
public interface HasParameters {

    /**
     * @return Returns the names of the parameters used by this gadget.
     */
    public List<String> getParameters();
}
