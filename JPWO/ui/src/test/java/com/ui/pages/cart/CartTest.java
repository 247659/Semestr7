package com.ui.pages.cart;

import com.ui.pages.product.SingleProductPage;
import com.ui.pages.user.LoginPage;
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
    private SingleProductPage singleProductPage;
    private CartPage cartPage;

    @BeforeEach
    void setUp() {
        driver = new ChromeDriver();
        driver.manage().window().maximize();
        driver.get("http://localhost:4200/auth/login");
        LoginPage loginPage = new LoginPage(driver);
        loginPage.loginAs("customer3@practicesoftwaretesting.com", "pass123");
        productPage = new ProductPage(driver);
        cartPage = new CartPage(driver);
        singleProductPage = new SingleProductPage(driver);
    }

    @Test
    void shouldAddProductToCart() {
        driver.get("http://localhost:4200/");
        productPage.openFirstProduct();
        String productName = singleProductPage.getProductName();
        singleProductPage.addToCart();
        assertEquals(1, singleProductPage.getCartCount(), "Cart should not be empty after adding a product");
        singleProductPage.openCartPage();
        assertTrue(cartPage.containsProduct(productName), "Cart does not contain the expected product");
    }


    @AfterEach
    void tearDown() {
        driver.quit();
    }
}
