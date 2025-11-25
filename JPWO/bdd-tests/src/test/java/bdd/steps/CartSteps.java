package bdd.steps;

import io.cucumber.java.en.Then;
import io.cucumber.java.en.When;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class CartSteps {


    @When("I add the product to the cart")
    public void iAddTheProductToTheCart() {
        SharedClass.singleProductPage.addToCart();
        assertEquals(1, SharedClass.singleProductPage.getCartCount(),
                "Cart count should be 1 after adding a product");
    }

    @Then("the cart should contain this product")
    public void theCartShouldContainThisProduct() {
        SharedClass.singleProductPage.openCartPage();
        assertTrue(SharedClass.cartPage.containsProduct(SharedClass.selectedProductName),
                "Cart does not contain the expected product");
    }
}
