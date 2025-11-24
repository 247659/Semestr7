package bdd.steps;

import bdd.utils.DriverFactory;
import io.cucumber.java.After;
import io.cucumber.java.Before;

public class Hooks {

    @Before
    public void beforeScenario() {
        DriverFactory.getDriver();
    }

    @After
    public void tearDown() {
        DriverFactory.quit();
    }
}
