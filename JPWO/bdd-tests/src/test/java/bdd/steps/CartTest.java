package bdd.steps;

import bdd.pages.cart.CartPage;
import bdd.pages.product.ProductPage;
import bdd.pages.product.SingleProductPage;
import bdd.pages.user.LoginPage;
import bdd.utils.DriverFactory;
import io.cucumber.java.en.Given;
import io.cucumber.java.en.Then;
import io.cucumber.java.en.When;
import org.openqa.selenium.WebDriver;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class CartTest {
    private final WebDriver driver = DriverFactory.getDriver();

    private ProductPage productPage;
    private SingleProductPage singleProductPage;
    private CartPage cartPage;
    private LoginPage loginPage;

    private String selectedProductName;

    @Given("I am logged in as {string} with password {string}")
    public void iAmLoggedIn(String email, String password) {
        driver.get("http://localhost:4200/auth/login");
        loginPage = new LoginPage(driver);
        loginPage.loginAs(email, password);

        productPage = new ProductPage(driver);
        singleProductPage = new SingleProductPage(driver);
        cartPage = new CartPage(driver);
    }

    @Given("I am on the home page")
    public void iAmOnHomePage() {
        driver.get("http://localhost:4200/");
    }

    @When("I open the first product")
    public void iOpenTheFirstProduct() {
        productPage.openFirstProduct();
        selectedProductName = singleProductPage.getProductName();
    }

    @When("I add the product to the cart")
    public void iAddTheProductToTheCart() {
        singleProductPage.addToCart();
        assertEquals(1, singleProductPage.getCartCount(),
                "Cart count should be 1 after adding a product");
    }

    @Then("the cart should contain this product")
    public void theCartShouldContainThisProduct() {
        singleProductPage.openCartPage();
        assertTrue(cartPage.containsProduct(selectedProductName),
                "Cart does not contain the expected product");
    }
}
