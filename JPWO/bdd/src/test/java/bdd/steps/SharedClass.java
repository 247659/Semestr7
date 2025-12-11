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
import org.openqa.selenium.WebDriver;

public class SharedClass {

    public static WebDriver driver;
    public static ProductPage productPage;
    public static SingleProductPage singleProductPage;
    public static CartPage cartPage;
    public static FavoritePage favoritePage;
    public static ContactPage contactPage;
    public static MessagesPage messagesPage;
    public static LoginPage loginPage;
    public static AccountPage accountPage;
    public static RegistrationPage registrationPage;
    public static CategoryAdminPage categoryAdminPage;
    public static AddCategoryPage addCategoryPage;

    public static String selectedProductName;
}
