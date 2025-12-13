package bdd.steps;

import bdd.pages.account.AccountPage;
import bdd.pages.cart.CartPage;
import bdd.pages.category.AddCategoryPage;
import bdd.pages.category.CategoryAdminPage;
import bdd.pages.contact.ContactPage;
import bdd.pages.contact.MessagesPage;
import bdd.pages.favorites.FavoritePage;
import bdd.pages.product.ProductPage;
import bdd.pages.product.SingleProductPage;
import bdd.pages.user.LoginPage;
import bdd.pages.user.RegistrationPage;
import bdd.utils.DriverFactory;
import io.cucumber.java.After;
import io.cucumber.java.Before;

public class Hooks {

    @Before
    public void beforeScenario() {
        SharedClass.driver = DriverFactory.getDriver();
        SharedClass.productPage = new ProductPage(SharedClass.driver);
        SharedClass.singleProductPage = new SingleProductPage(SharedClass.driver);
        SharedClass.cartPage = new CartPage(SharedClass.driver);
        SharedClass.favoritePage = new FavoritePage(SharedClass.driver);
        SharedClass.contactPage = new ContactPage(SharedClass.driver);
        SharedClass.messagesPage = new MessagesPage(SharedClass.driver);
        SharedClass.loginPage = new LoginPage(SharedClass.driver);
        SharedClass.registrationPage = new RegistrationPage(SharedClass.driver);
        SharedClass.accountPage = new AccountPage(SharedClass.driver);
        SharedClass.categoryAdminPage = new CategoryAdminPage(SharedClass.driver);
        SharedClass.addCategoryPage = new AddCategoryPage(SharedClass.driver);
    }

    @After
    public void tearDown() {
        DriverFactory.quit();
    }
}
