package bdd.steps;

import io.cucumber.java.en.Given;
import io.cucumber.java.en.When;

public class OpeningProductSteps {

    @Given("I open the home page")
    public void iAmOnHomePage() {
        SharedClass.driver.get("http://localhost:4200/");
    }

    @When("I open the first product")
    public void iOpenTheFirstProduct() {
        SharedClass.productPage.openFirstProduct();
        SharedClass.selectedProductName = SharedClass.singleProductPage.getProductName();
    }

}
