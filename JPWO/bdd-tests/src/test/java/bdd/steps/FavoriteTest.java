//package bdd.steps;
//
//import bdd.pages.favorites.FavoritePage;
//import bdd.pages.product.ProductPage;
//import bdd.pages.product.SingleProductPage;
//import bdd.pages.user.LoginPage;
//import io.cucumber.java.en.Given;
//import io.cucumber.java.en.Then;
//import io.cucumber.java.en.When;
//import org.openqa.selenium.WebDriver;
//import org.openqa.selenium.chrome.ChromeDriver;
//
//import static org.junit.jupiter.api.Assertions.assertTrue;
//
//public class FavoriteTest {
//
//    private WebDriver driver;
//    private ProductPage productPage;
//    private SingleProductPage singleProductPage;
//    private FavoritePage favoritePage;
//    private String productName;
//
//    @Given("I am logged in as {string} with password {string}")
//    public void iAmLoggedIn(String email, String password) {
//        driver = new ChromeDriver();
//        driver.manage().window().maximize();
//        driver.get("http://localhost:4200/auth/login");
//
//        LoginPage loginPage = new LoginPage(driver);
//        loginPage.loginAs(email, password);
//
//        productPage = new ProductPage(driver);
//        singleProductPage = new SingleProductPage(driver);
//        favoritePage = new FavoritePage(driver);
//    }
//
//    @When("I open the home page")
//    public void iOpenHomePage() {
//        driver.get("http://localhost:4200/");
//    }
//
//    @When("I open the first product")
//    public void iOpenFirstProduct() {
//        productPage.openFirstProduct();
//        productName = singleProductPage.getProductName();
//    }
//
//    @When("I add the product to favorites")
//    public void iAddProductToFavorites() {
//        singleProductPage.addToFavorites();
//    }
//
//    @Then("the favorites page should contain that product")
//    public void favoritesPageShouldContainProduct() {
//        favoritePage.open();
//        assertTrue(favoritePage.containsProduct(productName), "Favorites does not contain the expected product");
//        driver.quit();
//    }
//}
