package com.ui.pages.cart;

import com.ui.pages.contact.ContactPage;
import com.ui.pages.user.LoginPage;
import com.ui.pages.contact.MessagesPage;
import com.ui.pages.product.ProductPage;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeDriver;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class CartTest {
    private WebDriver driver;
    private ProductPage productPage;
    private CartPage cartPage;
    private LoginPage loginPage;

    @BeforeEach
    void setUp() {
        driver = new ChromeDriver();
        driver.manage().window().maximize();
        driver.get("http://localhost:4200/auth/login");
        loginPage = new LoginPage(driver);
        loginPage.loginAs("customer3@practicesoftwaretesting.com", "pass123");
        productPage = new ProductPage(driver);
        cartPage = new CartPage(driver);
    }

    @Test
    void shouldAddProductToCart() {
        productPage.open("01K8V39GY1R47X8K89E7DJ0HTX");

        productPage.addToCart();
        assertEquals(1, productPage.getCartCount(), "Cart should not be empty after adding a product");
        productPage.openCartPage();
        assertTrue(cartPage.containsProduct("Flat-Head Wood Screws"), "Cart does not contain the expected product");
    }


    @AfterEach
    void tearDown() {
        driver.quit();
    }
}
