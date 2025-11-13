package com.ui.pages.product;

import com.ui.pages.user.LoginPage;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeDriver;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class ProductTest {

    private WebDriver driver;
    private AddProductPage addProductPage;
    private ProductAdminPage productAdminPage;
    private ProductPage productPage;

    @BeforeEach
    void setUp() {
        driver = new ChromeDriver();
        driver.manage().window().maximize();
        driver.get("http://localhost:4200/auth/login");
        LoginPage loginPage = new LoginPage(driver);
        loginPage.loginAsAdmin("admin@practicesoftwaretesting.com", "welcome01");
        addProductPage = new AddProductPage(driver);
        productAdminPage = new ProductAdminPage(driver);
        productPage = new ProductPage(driver);
    }

    @Test
    void shouldAddNewProduct() throws InterruptedException {
        String uniqueProductName = "New Product " + System.currentTimeMillis();
        addProductPage.open();
        addProductPage.fillForm(uniqueProductName, "New product's description", 100, 29.99,
                "ForgeFlex Tools", "Hand Tools", "Hammer");
        productAdminPage.open();
        productAdminPage.search(uniqueProductName);
        String addedProductName = productAdminPage.getProductName(uniqueProductName);
        String addedProductPrice = productAdminPage.getProductPrice(uniqueProductName);

        assertEquals(uniqueProductName, addedProductName, "Product name does not match");
        assertEquals("$29.99", addedProductPrice, "Product price does not match");
    }

    @Test
    void shouldReturnMatchingProductsOnSearch() {
        driver.get("http://localhost:4200/");
        productPage.search("screws");

        List<String> names = productPage.getVisibleProductNames();

        assertTrue(names.stream().allMatch(n -> n.toLowerCase().contains("screws")),
                "Search returned unrelated products");
    }

    @AfterEach
    void tearDown() {
        driver.quit();
    }
}
