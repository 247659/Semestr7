package bdd.steps;

import io.cucumber.java.en.Then;
import io.cucumber.java.en.When;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class FavoriteSteps {

    @When("I add the product to favorites")
    public void iAddProductToFavorites() {
        SharedClass.singleProductPage.addToFavorites();
    }

    @Then("the favorites page should contain that product")
    public void favoritesPageShouldContainProduct() {
        SharedClass.favoritePage.open();
        assertTrue(SharedClass.favoritePage.containsProduct(SharedClass.selectedProductName), "Favorites does not contain the expected product");
    }

    @When("I remove the product from favorites")
    public void iRemoveProductFromFavorites() throws InterruptedException {
        SharedClass.favoritePage.open();
        SharedClass.favoritePage.removeFromFavorites();
    }

    @Then("the favorites page should not contain that product")
    public void favoritesPageShouldNotContainProduct() {
        assertFalse(SharedClass.favoritePage.containsProduct(SharedClass.selectedProductName), "Favorites still contains the product");
    }
}
