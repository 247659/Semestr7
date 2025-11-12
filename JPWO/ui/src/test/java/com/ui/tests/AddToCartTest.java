package com.ui.tests;

import com.ui.pages.CartPage;
import com.ui.pages.ProductPage;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeDriver;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class AddToCartTest {
    private WebDriver driver;
    private ProductPage productPage;
    private CartPage cartPage;

    @BeforeEach
    void setUp() {
        driver = new ChromeDriver();
        driver.manage().window().maximize();
        productPage = new ProductPage(driver);
        cartPage = new CartPage(driver);
    }

    @Test
    void shouldAddProductToCart() {
        productPage.open("01K8V39GY1R47X8K89E7DJ0HTX"); // np. /products/1

        productPage.addToCart();

        assertTrue(productPage.getCartCount() == 1, "Cart should not be empty after adding a product");

        productPage.openCartPage();
        assertTrue(cartPage.containsProduct("Flat-Head Wood Screws"), "Cart does not contain the expected product");
    }

    @AfterEach
    void tearDown() {
        driver.quit();
    }
}
